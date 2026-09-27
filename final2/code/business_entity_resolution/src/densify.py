"""Build a test-like training split ("train_dense") with the test set's distractor density.

Why: the provided test files contain 5.75 S2/S3 records per S1 entity, train only 4.68.
With the same number of true matches per S1 (~3.46), test has ~2.29 distractor records
per S1 against ~1.22 in train, i.e. about twice as many look-alike records that must be
rejected.  A model trained and tuned on train is therefore too willing to match on test.

How: remove a fraction x of the train S1 entities (deterministic hash of the id).  Their
S2/S3 records stay in the data but lose their owner, so they become distractors, exactly
like a business that appears in S2/S3 but not in the reference source.  x solves

    (d_train + m * x) / (1 - x) = d_test,   m = true matches per S1,
    d = (S2/S3 records per S1) - m          (distractors per S1)

using only record counts of the provided files (no labels, no model output).  All folds
are densified alike; fold buckets are unchanged (records keep their cluster bucket).

Output: WORK_DIR/train_dense/records.parquet and emb.npy (row subset of train).

    python src/densify.py
"""
import json
import time
import zlib

import numpy as np
import polars as pl

from config import DENSIFY_FRAC, split_dir


def target_fraction(train: pl.DataFrame, test_counts: tuple) -> dict:
    n_s1 = train.filter(pl.col("src") == 1).height
    n_r = train.filter(pl.col("src") != 1).height
    m = train["true_s1"].is_not_null().sum() / n_s1
    d_train = n_r / n_s1 - m
    t_s1, t_r = test_counts
    d_test = t_r / t_s1 - m
    x = max(0.0, (d_test - d_train) / (m + d_test))
    return {"matches_per_s1": m, "distractors_per_s1_train": d_train, "distractors_per_s1_test": d_test,
            "records_per_s1_train": n_r / n_s1, "records_per_s1_test": t_r / t_s1, "remove_fraction": x}


def main():
    t = time.time()
    train = pl.read_parquet(split_dir("train") / "records.parquet")
    test = pl.read_parquet(split_dir("test") / "records.parquet", columns=["src"])
    test_counts = (test.filter(pl.col("src") == 1).height, test.filter(pl.col("src") != 1).height)
    info = target_fraction(train, test_counts)
    x = DENSIFY_FRAC if DENSIFY_FRAC is not None else info["remove_fraction"]
    info["remove_fraction_used"] = x
    ids = train["entity_id"].to_list()
    drop = np.array([s.startswith("S1-") and zlib.crc32(b"densify:" + s.encode()) % 100_000 < x * 100_000
                     for s in ids])
    removed = set(np.array(ids, dtype=object)[drop].tolist())
    keep = ~drop
    dense = train.filter(pl.Series(keep)).with_columns(
        pl.when(pl.col("true_s1").is_in(pl.Series(list(removed), dtype=pl.Utf8).implode()))
        .then(None).otherwise(pl.col("true_s1")).alias("true_s1"))
    out = split_dir("train_dense")
    dense.write_parquet(out / "records.parquet")
    emb = np.load(split_dir("train") / "emb.npy", mmap_mode="r")
    np.save(out / "emb.npy", emb[np.nonzero(keep)[0]])
    r = dense.filter(pl.col("src") != 1)
    n_s1 = dense.filter(pl.col("src") == 1).height
    info.update({
        "s1_removed": int(drop.sum()), "s1_kept": n_s1,
        "distractor_share_train": float(train.filter(pl.col("src") != 1)["true_s1"].is_null().mean()),
        "distractor_share_dense": float(r["true_s1"].is_null().mean()),
        "records_per_s1_dense": r.height / n_s1,
    })
    for k, v in info.items():
        print(f"[densify] {k:28s} {v:,.4f}" if isinstance(v, float) else f"[densify] {k:28s} {v:,}")
    (out / "densify.json").write_text(json.dumps(info, indent=2))
    print(f"[densify] wrote {out} ({time.time()-t:.0f}s)")


if __name__ == "__main__":
    main()
