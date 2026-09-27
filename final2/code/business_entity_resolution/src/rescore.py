"""Step 7b - stage-2 re-scorer: cluster-context features on top of the matcher (stacking).

The pairwise matcher scores each (S1, S2/S3) pair on its own.  Entity resolution has
structure it cannot see: an S2/S3 record belongs to at most one S1, and the true members
of one S1 cluster resemble each other.  Stage 2 re-scores every pair with the stage-1
probability p plus context features derived from p over the whole candidate graph:

  record side   best p of the S2/S3 record, gap to it, runner-up p, rank of this pair,
                number of confident (p >= 0.5) S1 for the record, is-best flag
  S1 side       expected cluster size (sum of p), confident members overall and from the
                same source, rank of this pair inside the S1 (overall / same source)
  siblings      agreement of the record with the S1's other confident members (up to 3,
                p >= 0.9): max embedding cosine, name token-set, address token-set - the
                decisive signal for records WITHOUT an address, whose true S1 is often
                ambiguous by name alone
  base          the strongest stage-1 inputs (cos, name/address similarities, numbers,
                missing address, name frequency)

Training uses the matcher's OUT-OF-FOLD p and the same fold split (stage-2 model A is
trained on fold A and scores fold B, and vice versa), so there is no leakage.  At test
time p = mean of the two matchers and p2 = mean of the two stage-2 models.  The decision
rule is re-tuned on out-of-fold p2; stage 2 is only used if it beats stage 1.

    python src/rescore.py --split train_dense          # train + evaluate (writes matcher*/stage2/)
"""
import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz

from config import ENCODER_BUCKETS, FOLD_A_BUCKETS, FOLD_B_BUCKETS, OMP_THREADS, split_dir
from decision import decide, macro_f05
from features import str_sims
from train_matcher import MATCHER_DIR, fit, predict, search_rule

STAGE2_DIR = MATCHER_DIR / "stage2"
BASE_COLS = ["cos", "rank", "margin_r", "p1", "name_tsort", "core_eq", "addr_tset", "num_jacc", "num_best_sim",
             "addr_missing_r", "addr_missing_s1", "s1_name_freq", "src", "ce"]  # "ce" only if present
CONF_SIB, N_SIB = 0.9, 3


def context_features(pairs: pl.DataFrame, recs: pl.DataFrame, emb) -> pl.DataFrame:
    """pairs: r_idx, s1_idx, src, p (+ base columns).  Returns pairs + stage-2 features."""
    t = time.time()
    df = pairs.with_columns(
        pl.col("p").max().over("r_idx").alias("pmax_r"),
        pl.col("p").rank("ordinal", descending=True).over("r_idx").alias("prank_r"),
        (pl.col("p") >= 0.5).sum().over("r_idx").alias("nconf_r"),
        pl.col("p").sum().over("s1_idx").alias("psum_s1"),
        (pl.col("p") >= 0.9).sum().over("s1_idx").alias("nconf_s1"),
        (pl.col("p") >= 0.9).sum().over(["s1_idx", "src"]).alias("nconf_s1_src"),
        pl.col("p").rank("ordinal", descending=True).over("s1_idx").alias("prank_s1"),
        pl.col("p").rank("ordinal", descending=True).over(["s1_idx", "src"]).alias("prank_s1_src"),
    )
    second = df.filter(pl.col("prank_r") == 2).select("r_idx", pl.col("p").alias("p2nd_r"))
    df = df.join(second, on="r_idx", how="left").with_columns(
        pl.col("p2nd_r").fill_null(0.0),
        (pl.col("p") - pl.col("pmax_r")).alias("pgap_r"),
        (pl.col("prank_r") == 1).cast(pl.Float32).alias("is_best_r"),
        (pl.col("psum_s1") - pl.col("p")).alias("psum_s1_others"),
    )
    # ---- siblings: up to N_SIB most confident other members of the same S1
    sib = df.filter(pl.col("p") >= CONF_SIB).sort("p", descending=True) \
        .group_by("s1_idx", maintain_order=True).head(N_SIB + 1) \
        .select("s1_idx", pl.col("r_idx").alias("sib_idx"))
    ps = df.select("r_idx", "s1_idx").join(sib, on="s1_idx").filter(pl.col("r_idx") != pl.col("sib_idx"))
    ps = ps.group_by(["r_idx", "s1_idx"], maintain_order=True).head(N_SIB)
    a, b = ps["r_idx"].to_numpy(), ps["sib_idx"].to_numpy()
    cos = np.empty(len(a), dtype=np.float32)
    for i in range(0, len(a), 2_000_000):
        cos[i:i + 2_000_000] = np.einsum("ij,ij->i", emb[a[i:i + 2_000_000]].astype(np.float32),
                                         emb[b[i:i + 2_000_000]].astype(np.float32))
    core, addr = recs["name_core"].to_numpy(), recs["addr"].to_numpy()
    sims = str_sims(core[a].tolist(), core[b].tolist(), "sib_name", {"tset": fuzz.token_set_ratio})
    sims.update(str_sims(addr[a].tolist(), addr[b].tolist(), "sib_addr", {"tset": fuzz.token_set_ratio}))
    ps = ps.with_columns(pl.Series("sib_cos", cos), *[pl.Series(k, v) for k, v in sims.items()])
    agg = ps.group_by(["r_idx", "s1_idx"]).agg(
        pl.col("sib_cos").max().alias("sib_cos_max"), pl.col("sib_cos").mean().alias("sib_cos_mean"),
        pl.col("sib_name_tset").max().alias("sib_name_max"), pl.col("sib_addr_tset").max().alias("sib_addr_max"),
        pl.len().alias("n_sib"))
    df = df.join(agg, on=["r_idx", "s1_idx"], how="left").with_columns(pl.col("n_sib").fill_null(0))
    print(f"[stage2] context features for {df.height:,} pairs ({time.time()-t:.0f}s)", flush=True)
    return df


def stage2_columns(df):
    return [c for c in df.columns if c not in ("r_idx", "s1_idx", "y", "bucket", "r_id", "s1_id", "country")]


def load_pairs(split: str, p: np.ndarray = None):
    d = split_dir(split)
    feats = pl.read_parquet(d / "features.parquet")
    recs = pl.read_parquet(d / "records.parquet")
    ids = recs["entity_id"].to_numpy()
    keep = ["r_idx", "s1_idx"] + [c for c in BASE_COLS if c in feats.columns] + \
        [c for c in ("y", "bucket") if c in feats.columns]
    pairs = feats.select(keep).with_columns(
        pl.Series("r_id", ids[feats["r_idx"].to_numpy()]), pl.Series("s1_id", ids[feats["s1_idx"].to_numpy()]),
        pl.Series("country", recs["country"].to_numpy()[feats["r_idx"].to_numpy()]))
    if p is None:  # train-like split: out-of-fold matcher probabilities
        oof = pl.read_parquet(MATCHER_DIR / f"oof_{split}.parquet", columns=["s1_id", "r_id", "p"])
        pairs = pairs.join(oof, on=["s1_id", "r_id"])
    else:
        pairs = pairs.with_columns(pl.Series("p", p))
    emb = np.load(d / "emb.npy", mmap_mode="r")
    return context_features(pairs, recs, emb), recs


def train(split: str, max_rows: int, rounds: int):
    t = time.time()
    df, recs = load_pairs(split)
    cols = stage2_columns(df)
    in_a, in_b = pl.col("bucket").is_in(list(FOLD_A_BUCKETS)), pl.col("bucket").is_in(list(FOLD_B_BUCKETS))
    fa, fb = df.filter(in_a), df.filter(in_b)
    enc = df.filter(pl.col("bucket").is_in(list(ENCODER_BUCKETS)))
    ma = fit(fa, cols, max_rows, rounds, desc="stage-2 model A")
    mb = fit(fb, cols, max_rows, rounds, desc="stage-2 model B")
    oof = pl.concat([
        fb.with_columns(pl.Series("p2", predict([ma], fb, cols))),
        fa.with_columns(pl.Series("p2", predict([mb], fa, cols))),
        enc.with_columns(pl.Series("p2", predict([ma, mb], enc, cols))),
    ])
    s1 = recs.filter((pl.col("src") == 1) & pl.col("bucket").is_in(list(FOLD_A_BUCKETS + FOLD_B_BUCKETS)))
    truth = recs.filter(pl.col("true_s1").is_not_null()).select(
        pl.col("true_s1").alias("s1_id"), pl.col("entity_id").alias("r_id")) \
        .join(s1.select(pl.col("entity_id").alias("s1_id")), on="s1_id")
    stage1 = search_rule(oof.select("s1_id", "r_id", "p", "cos"), truth, s1["entity_id"], "stage 1 (reference)")
    stage2 = search_rule(oof.select("s1_id", "r_id", pl.col("p2").alias("p"), "cos"), truth, s1["entity_id"],
                         "stage 2")
    report = {"split": split, "stage1": {"rule": stage1[0], "threshold": stage1[1], "macro_f05": stage1[2]},
              "stage2": {"rule": stage2[0], "threshold": stage2[1], "macro_f05": stage2[2]},
              "gain": stage2[2] - stage1[2], "per_country": {}}
    p2 = oof.select("s1_id", "r_id", pl.col("p2").alias("p"), "cos", "country")
    for country in sorted(s1["country"].unique().to_list()):
        s1c = s1.filter(pl.col("country") == country)["entity_id"]
        tc = truth.filter(pl.col("s1_id").is_in(s1c.implode()))
        f1 = macro_f05(decide(oof.filter(pl.col("country") == country).select("s1_id", "r_id", "p", "cos"),
                              stage1[0], stage1[1]), tc, s1c)
        f2 = macro_f05(decide(p2.filter(pl.col("country") == country), stage2[0], stage2[1]), tc, s1c)
        report["per_country"][country] = {"stage1": f1, "stage2": f2}
        print(f"[stage2]   {country}: stage 1 {f1:.5f} -> stage 2 {f2:.5f}")
    print(f"[stage2] {split}: stage 1 {stage1[2]:.5f} -> stage 2 {stage2[2]:.5f}  (gain {report['gain']:+.5f})")
    STAGE2_DIR.mkdir(parents=True, exist_ok=True)
    ma.save_model(str(STAGE2_DIR / "model_A.txt"), num_iteration=ma.best_iteration)
    mb.save_model(str(STAGE2_DIR / "model_B.txt"), num_iteration=mb.best_iteration)
    imp = sorted(zip(cols, ma.feature_importance("gain")), key=lambda x: -x[1])
    report["top_features_gain"] = [(c, round(float(g), 1)) for c, g in imp[:20]]
    (STAGE2_DIR / "report.json").write_text(json.dumps(report, indent=2))
    print(f"[stage2] saved to {STAGE2_DIR} ({time.time()-t:.0f}s)")


def apply(split: str, p: np.ndarray) -> pl.DataFrame:
    """Test time: stage-2 probabilities for every candidate pair (feature row order kept)."""
    df, _ = load_pairs(split, p)
    models = [lgb.Booster(model_file=str(STAGE2_DIR / f"model_{k}.txt")) for k in ("A", "B")]
    x = df.select(models[0].feature_name()).to_numpy()
    p2 = np.mean([m.predict(x, num_threads=OMP_THREADS) for m in models], axis=0)
    return df.select("s1_id", "r_id", "cos").with_columns(pl.Series("p", p2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train_dense")
    ap.add_argument("--max-rows", type=int, default=12_000_000)
    ap.add_argument("--rounds", type=int, default=3000)
    args = ap.parse_args()
    train(args.split, args.max_rows, args.rounds)


if __name__ == "__main__":
    main()
