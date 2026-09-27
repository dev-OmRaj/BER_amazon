"""Step 4 - candidate generation stage 1: cheap learned prefilter (blocking cascade).

The kNN stage (blocking.py) is tuned for recall and keeps ~11-33 candidates per S1
entity.  Most of them are obviously wrong.  A small LightGBM model on 21 cheap features
(embedding similarity/rank/competition, 4 rapidfuzz similarities, number-set overlap,
missing-address flags, source) scores every kNN pair; pairs below a threshold t1 are
discarded.  t1 is the lowest score that still keeps PREFILTER_RECALL (99.95%) of the
true pairs found by the kNN stage, measured on out-of-fold scores.

The surviving pairs are the FINAL candidate set: exactly the pairs the matcher scores,
written to output/candidate_pairs.tsv.  The stage-1 probability p1 is passed on to the
matcher as a feature (out-of-fold on train, so no leakage).

    python src/prefilter.py --split train          # cross-fit stage-1 models, choose t1, filter
    python src/prefilter.py --split test           # apply the saved models / t1
    python src/prefilter.py --split train_dense --reuse
        # score a train-like split with the SAVED models, out-of-fold (model A scores fold B
        # and vice versa) and filter with the saved t1 - used to evaluate existing models
"""
import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz

from blocking import KNN_FILE
from blocking_stats import report
from config import (FOLD_A_BUCKETS, FOLD_B_BUCKETS, OMP_THREADS, PREFILTER_RECALL, PREFILTER_TAG, SEED, model_dir,
                    split_dir)
from features import add_labels, embedding_features, number_overlap, str_sims
from train_matcher import TqdmCallback

PREFILTER_DIR = model_dir("prefilter", PREFILTER_TAG)
PARAMS = dict(
    objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=500,
    feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, max_bin=127,
    verbose=-1, seed=SEED, num_threads=OMP_THREADS,
)


def cheap_features(split: str) -> pl.DataFrame:
    d = split_dir(split)
    recs = pl.read_parquet(d / "records.parquet")
    cand = pl.read_parquet(d / KNN_FILE)
    a, b = cand["r_idx"].to_numpy(), cand["s1_idx"].to_numpy()
    feats = embedding_features(cand)
    name, core, addr = (recs[c].to_numpy() for c in ("name", "name_core", "addr"))
    feats.update(str_sims(name[a].tolist(), name[b].tolist(), "name", {"tset": fuzz.token_set_ratio}))
    feats.update(str_sims(core[a].tolist(), core[b].tolist(), "core", {"ratio": fuzz.ratio}))
    feats.update(str_sims(addr[a].tolist(), addr[b].tolist(), "addr",
                          {"tset": fuzz.token_set_ratio, "ratio": fuzz.ratio}))
    feats.update(number_overlap(recs, a, b)[3])
    missing = recs["addr_missing"].to_numpy()
    feats["addr_missing_r"] = missing[a].astype(np.float32)
    feats["addr_missing_s1"] = missing[b].astype(np.float32)
    feats["src"] = recs["src"].to_numpy()[a].astype(np.float32)
    df = pl.DataFrame({"r_idx": a, "s1_idx": b, **feats})
    df = df.with_columns([(pl.col(c) - pl.col(c).max().over("r_idx")).alias(f"{c}_dr")
                          for c in ("name_tset", "addr_tset")])
    df = add_labels(df, recs, a, b)
    return cand, df


def feature_cols(df):
    return [c for c in df.columns if c not in ("r_idx", "s1_idx", "y", "bucket")]


def fit(df, cols, desc, max_rows=6_000_000, rounds=600):
    r = df.select("r_idx").unique()
    if df.height > max_rows:
        r = r.sample(fraction=max_rows / df.height, seed=SEED)
    hold = r.sample(fraction=0.05, seed=SEED + 1)
    tr = df.join(r, on="r_idx").join(hold, on="r_idx", how="anti")
    va = df.join(hold, on="r_idx")
    dtr = lgb.Dataset(tr.select(cols).to_numpy(), tr["y"].to_numpy(), feature_name=cols)
    dva = lgb.Dataset(va.select(cols).to_numpy(), va["y"].to_numpy(), reference=dtr)
    progress = TqdmCallback(rounds, desc)
    m = lgb.train(PARAMS, dtr, rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(30, verbose=False), progress])
    progress.bar.close()
    return m


def write_survivors(split, cand, p1, t1):
    keep = p1 >= t1
    out = cand.with_columns(pl.Series("p1", p1.astype(np.float32))).filter(pl.Series(keep))
    out.write_parquet(split_dir(split) / "candidates.parquet")
    print(f"[prefilter] {split}: kept {out.height:,} of {cand.height:,} kNN pairs ({out.height / cand.height:.1%}) "
          f"at t1={t1:.5f}")
    report(split, cand, "knn")
    report(split, out, "final")


def cross_scores(ma, mb, df, cols):
    """Out-of-fold stage-1 scores: model A scores fold B, B scores A, encoder buckets get the mean."""
    x = df.select(cols).to_numpy()
    bucket = df["bucket"].to_numpy()
    pa, pb = ma.predict(x, num_threads=OMP_THREADS), mb.predict(x, num_threads=OMP_THREADS)
    is_a = np.isin(bucket, FOLD_A_BUCKETS)
    is_b = np.isin(bucket, FOLD_B_BUCKETS)
    return np.where(is_b, pa, np.where(is_a, pb, 0.5 * (pa + pb))), is_a | is_b


def reuse(split):
    t = time.time()
    info = json.loads((PREFILTER_DIR / "prefilter.json").read_text())
    ma, mb = (lgb.Booster(model_file=str(PREFILTER_DIR / f"model_{k}.txt")) for k in ("A", "B"))
    cand, df = cheap_features(split)
    p1, ev = cross_scores(ma, mb, df, ma.feature_name())
    y = df["y"].to_numpy()
    print(f"[prefilter] {split} (saved models, out-of-fold): true kNN pairs kept "
          f"{(p1[ev & (y == 1)] >= info['t1']).mean():.5f} at saved t1={info['t1']:.5f}")
    write_survivors(split, cand, p1, info["t1"])
    print(f"[prefilter] done ({time.time()-t:.0f}s)")


def train(split="train"):
    t = time.time()
    cand, df = cheap_features(split)
    cols = feature_cols(df)
    print(f"[prefilter] {split}: {df.height:,} kNN pairs, {len(cols)} cheap features ({time.time()-t:.0f}s)")
    in_a, in_b = pl.col("bucket").is_in(list(FOLD_A_BUCKETS)), pl.col("bucket").is_in(list(FOLD_B_BUCKETS))
    ma = fit(df.filter(in_a), cols, "stage-1 model A")
    mb = fit(df.filter(in_b), cols, "stage-1 model B")
    p1, ev = cross_scores(ma, mb, df, cols)
    y = df["y"].to_numpy()
    t1 = float(np.quantile(p1[ev & (y == 1)], 1 - PREFILTER_RECALL))
    PREFILTER_DIR.mkdir(parents=True, exist_ok=True)
    ma.save_model(str(PREFILTER_DIR / "model_A.txt"))
    mb.save_model(str(PREFILTER_DIR / "model_B.txt"))
    info = {"t1": t1, "target_recall": PREFILTER_RECALL,
            "kept_fraction_oof": float((p1[ev] >= t1).mean()),
            "true_pairs_kept_oof": float((p1[ev & (y == 1)] >= t1).mean())}
    (PREFILTER_DIR / "prefilter.json").write_text(json.dumps(info, indent=2))
    print(f"[prefilter] {info}")
    write_survivors(split, cand, p1, t1)
    print(f"[prefilter] done ({time.time()-t:.0f}s)")


def apply(split):
    t = time.time()
    info = json.loads((PREFILTER_DIR / "prefilter.json").read_text())
    models = [lgb.Booster(model_file=str(PREFILTER_DIR / f"model_{k}.txt")) for k in ("A", "B")]
    cand, df = cheap_features(split)
    x = df.select(models[0].feature_name()).to_numpy()
    p1 = np.mean([m.predict(x, num_threads=OMP_THREADS) for m in models], axis=0)
    write_survivors(split, cand, p1, info["t1"])
    print(f"[prefilter] done ({time.time()-t:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--reuse", action="store_true", help="train-like split: evaluate the saved models")
    args = ap.parse_args()
    if not args.split.startswith("train"):
        apply(args.split)
    elif args.reuse:
        reuse(args.split)
    else:
        train(args.split)


if __name__ == "__main__":
    main()
