"""Step 5 - train the pairwise matcher (LightGBM) with 2-fold cross-fitting.

Folds are defined on S1 clusters (see config): model A is trained on fold-A pairs and
scores fold B, model B the reverse, so every validation score is out-of-fold.  The
decision rule/threshold is then chosen by maximising the challenge metric (macro F0.5
over all S1 entities of folds A+B, singletons included).  At test time the two models
are averaged.

    python src/train_matcher.py [--max-rows 12000000] [--transfer]

--transfer additionally trains on US only and evaluates on India (and vice versa):
a proxy for how well the model generalises to an unseen country (France).
"""
import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from tqdm import tqdm

from config import ENCODER_BUCKETS, FOLD_A_BUCKETS, FOLD_B_BUCKETS, SEED, WORK_DIR, split_dir
from decision import decide, macro_f05
from features import feature_columns
from io_utils import read_ground_truth

MATCHER_DIR = WORK_DIR / "matcher"
PARAMS = dict(
    objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=200,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    max_bin=255, verbose=-1, seed=SEED, num_threads=0,
)


class TqdmCallback:
    """LightGBM callback: one tqdm tick per boosting round, showing the validation loss."""

    def __init__(self, total, desc):
        self.bar = tqdm(total=total, desc=desc, unit="tree")

    def __call__(self, env):
        self.bar.update(1)
        if env.evaluation_result_list and (env.iteration + 1) % 10 == 0:
            self.bar.set_postfix(logloss=f"{env.evaluation_result_list[0][2]:.5f}", refresh=False)


def fit(df: pl.DataFrame, cols, max_rows, rounds, desc="lightgbm"):
    """Train on df (sub-sampled by S2/S3 record), early stopping on 5% held-out records."""
    r = df.select("r_idx").unique()
    if df.height > max_rows:
        r = r.sample(fraction=max_rows / df.height, seed=SEED)
    hold = r.sample(fraction=0.05, seed=SEED + 1)
    tr = df.join(r, on="r_idx").join(hold, on="r_idx", how="anti")
    va = df.join(hold, on="r_idx")
    dtr = lgb.Dataset(tr.select(cols).to_numpy(), tr["y"].to_numpy(), feature_name=cols, free_raw_data=True)
    dva = lgb.Dataset(va.select(cols).to_numpy(), va["y"].to_numpy(), reference=dtr)
    progress = TqdmCallback(rounds, desc)
    m = lgb.train(PARAMS, dtr, rounds, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(50, verbose=False), progress])
    progress.bar.close()
    print(f"[matcher]   trained on {tr.height:,} rows, best iteration {m.best_iteration}", flush=True)
    return m


def predict(models, df, cols):
    x = df.select(cols).to_numpy()
    return np.mean([m.predict(x, num_iteration=m.best_iteration) for m in models], axis=0)


def search_rule(pairs, truth, s1_ids, label=""):
    """Grid-search decision rule + threshold on the challenge metric."""
    best = ("threshold", 0.5, -1.0)
    grid = [("threshold", float(t)) for t in np.arange(0.2, 0.96, 0.05)] + \
           [("expected_f", float(t)) for t in np.arange(0.0, 0.91, 0.1)]
    for rule, t in tqdm(grid, desc=f"decision search {label}", unit="cfg"):
        f = macro_f05(decide(pairs, rule, t), truth, s1_ids)
        if f > best[2]:
            best = (rule, t, f)
    print(f"[matcher] {label} best rule={best[0]} t={best[1]:.2f}  macro F0.5={best[2]:.5f}")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=12_000_000, help="training rows per fold model")
    ap.add_argument("--rounds", type=int, default=5000)
    ap.add_argument("--transfer", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    d = split_dir("train")
    feats = pl.read_parquet(d / "features.parquet")
    cols = feature_columns(feats)
    recs = pl.read_parquet(d / "records.parquet", columns=["entity_id", "src", "country", "bucket"]) \
        .with_row_index("idx")
    ids = recs["entity_id"].to_numpy()
    feats = feats.with_columns(
        pl.Series("r_id", ids[feats["r_idx"].to_numpy()]),
        pl.Series("s1_id", ids[feats["s1_idx"].to_numpy()]),
        pl.Series("country", recs["country"].to_numpy()[feats["r_idx"].to_numpy()]),
    )
    print(f"[matcher] {feats.height:,} pairs, {len(cols)} features, positives {feats['y'].sum():,}")

    in_a = pl.col("bucket").is_in(list(FOLD_A_BUCKETS))
    in_b = pl.col("bucket").is_in(list(FOLD_B_BUCKETS))
    fa, fb = feats.filter(in_a), feats.filter(in_b)
    enc = feats.filter(pl.col("bucket").is_in(list(ENCODER_BUCKETS)))
    print("[matcher] model A (fold A -> scores fold B)")
    ma = fit(fa, cols, args.max_rows, args.rounds, desc="model A")
    print("[matcher] model B (fold B -> scores fold A)")
    mb = fit(fb, cols, args.max_rows, args.rounds, desc="model B")
    oof = pl.concat([
        fb.with_columns(pl.Series("p", predict([ma], fb, cols))),
        fa.with_columns(pl.Series("p", predict([mb], fa, cols))),
        enc.with_columns(pl.Series("p", predict([ma, mb], enc, cols))),
    ]).select("s1_id", "r_id", "p", "cos", "y", "country")
    print(f"[matcher] out-of-fold scoring done ({time.time()-t0:.0f}s)")
    MATCHER_DIR.mkdir(parents=True, exist_ok=True)
    oof.write_parquet(MATCHER_DIR / "oof.parquet")  # for error analysis

    # evaluation set: every S1 entity of folds A+B (singletons included)
    s1 = recs.filter((pl.col("src") == 1) & (in_a | in_b))
    truth = read_ground_truth().join(s1.select(pl.col("entity_id").alias("s1_id")), on="s1_id")
    rule, t, f = search_rule(oof, truth, s1["entity_id"], "overall")
    report = {"rule": rule, "threshold": t, "oof_macro_f05": f, "per_country": {}}
    for country in sorted(s1["country"].unique().to_list()):
        s1c = s1.filter(pl.col("country") == country)["entity_id"]
        pc = oof.filter(pl.col("country") == country)
        fc = macro_f05(decide(pc, rule, t), truth.filter(pl.col("s1_id").is_in(s1c.implode())), s1c)
        report["per_country"][country] = fc
        print(f"[matcher]   {country}: macro F0.5 = {fc:.5f}")
    # upper bound given blocking: perfect decisions on the candidate set
    perfect = oof.filter(pl.col("y") == 1).select("s1_id", "r_id")
    report["blocking_ceiling_f05"] = macro_f05(perfect, truth, s1["entity_id"])
    print(f"[matcher] ceiling with perfect matcher on these candidates: {report['blocking_ceiling_f05']:.5f}")

    if args.transfer:
        report["transfer"] = {}
        for src_c, dst_c in (("US", "India"), ("India", "US")):
            m = fit(fa.filter(pl.col("country") == src_c), cols, args.max_rows, args.rounds,
                    desc=f"transfer {src_c}")
            pc = fb.filter(pl.col("country") == dst_c)
            pc = pc.with_columns(pl.Series("p", predict([m], pc, cols))).select("s1_id", "r_id", "p", "cos")
            s1c = s1.filter((pl.col("country") == dst_c) & in_b)["entity_id"]
            tc = truth.filter(pl.col("s1_id").is_in(s1c.implode()))
            fc = macro_f05(decide(pc, rule, t), tc, s1c)
            report["transfer"][f"{src_c}->{dst_c}"] = fc
            print(f"[matcher] transfer {src_c} -> {dst_c}: macro F0.5 = {fc:.5f} (rule/t from main run)")

    MATCHER_DIR.mkdir(parents=True, exist_ok=True)
    ma.save_model(str(MATCHER_DIR / "model_A.txt"), num_iteration=ma.best_iteration)
    mb.save_model(str(MATCHER_DIR / "model_B.txt"), num_iteration=mb.best_iteration)
    imp = sorted(zip(cols, ma.feature_importance("gain")), key=lambda x: -x[1])
    report["top_features_gain"] = [(c, round(float(g), 1)) for c, g in imp[:25]]
    (MATCHER_DIR / "report.json").write_text(json.dumps(report, indent=2))
    print(f"[matcher] saved models + report to {MATCHER_DIR} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
