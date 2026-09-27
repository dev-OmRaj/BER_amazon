"""Step 1 - load raw TSVs, assign folds, learn dictionaries, normalise every record.

Output: WORK_DIR/{train,test}/records.parquet with one row per record of all 3 sources.

    python src/prepare.py            # both splits
    python src/prepare.py --sample 0.05   # dev run on 5% of S1 clusters (train only)
"""
import argparse
import time

import polars as pl
from tqdm import tqdm

import build_dicts
from config import pool, split_dir
from io_utils import bucket_of, read_ground_truth, read_sources
from text_norm import Normalizer, has_native

_NORM = None


def _norm_chunk(rows):
    out = []
    for name, addr, country in rows:
        n = _NORM.name(name)
        a = _NORM.address(addr, country)
        out.append((n["name"], n["name_core"], n["name_alt"], n["name_is_domain"], n["name_has_dba"],
                    has_native(name), a["addr"], a["addr_nums"]))
    return out


def normalise(records: pl.DataFrame, token_dict, comp_dict) -> pl.DataFrame:
    global _NORM
    _NORM = Normalizer(token_dict, comp_dict)
    rows = list(records.select("name_raw", "addr_raw", "country").iter_rows())
    step = 20000
    chunks = [rows[i:i + step] for i in range(0, len(rows), step)]
    res = []
    with pool() as p, tqdm(total=len(rows), desc="normalise", unit="rec", unit_scale=True) as bar:
        for part in p.imap(_norm_chunk, chunks):  # fork: workers inherit _NORM
            res.extend(part)
            bar.update(len(part))
    cols = ["name", "name_core", "name_alt", "name_is_domain", "name_has_dba", "name_native", "addr", "addr_nums"]
    normed = pl.DataFrame(res, schema=cols, orient="row")
    return pl.concat([records, normed], how="horizontal").with_columns(
        (pl.col("addr").str.len_chars() == 0).alias("addr_missing"))


def assign_buckets(records: pl.DataFrame, gt: pl.DataFrame) -> pl.DataFrame:
    """Bucket = hash of the S1 cluster id (true S1 id for matched records, own id otherwise)."""
    owner = gt.select(pl.col("r_id").alias("entity_id"), pl.col("s1_id").alias("true_s1"))
    records = records.join(owner, on="entity_id", how="left")
    key = pl.when(pl.col("src") == 1).then(pl.col("entity_id")) \
        .otherwise(pl.coalesce("true_s1", "entity_id"))
    records = records.with_columns(key.alias("_k"))
    return records.with_columns(bucket_of(records["_k"]).alias("bucket")).drop("_k")


def sample_clusters(records: pl.DataFrame, frac: float) -> pl.DataFrame:
    """Keep a random fraction of S1 clusters plus the same fraction of unmatched records."""
    keep = (pl.col("_k").hash(7) % 1000) < int(frac * 1000)
    key = pl.when(pl.col("src") == 1).then(pl.col("entity_id")).otherwise(
        pl.coalesce("true_s1", "entity_id"))
    return records.with_columns(key.alias("_k")).filter(keep).drop("_k")


def slice_by_regex(records: pl.DataFrame, pattern: str) -> pl.DataFrame:
    """Dev: keep clusters whose S1 address (or own address, if unmatched) matches `pattern`.

    Unlike random sampling this keeps the full local density of confusable records.
    """
    hit = records.filter(pl.col("addr_raw").fill_null("").str.contains(pattern))
    s1_keep = hit.filter(pl.col("src") == 1)["entity_id"]
    un_keep = hit.filter((pl.col("src") != 1) & pl.col("true_s1").is_null())["entity_id"]
    return records.filter(
        pl.col("entity_id").is_in(s1_keep.implode()) | pl.col("true_s1").is_in(s1_keep.implode())
        | pl.col("entity_id").is_in(un_keep.implode()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="train,test")
    ap.add_argument("--sample", type=float, default=None, help="dev: fraction of train clusters")
    ap.add_argument("--dev-regex", default=None, help="dev: keep clusters whose address matches")
    args = ap.parse_args()

    t = time.time()
    gt = read_ground_truth()
    train = assign_buckets(read_sources("train"), gt)
    if args.sample or args.dev_regex:
        train = sample_clusters(train, args.sample) if args.sample else slice_by_regex(train, args.dev_regex)
        gt = gt.filter(pl.col("s1_id").is_in(train.filter(pl.col("src") == 1)["entity_id"].implode()))
    print(f"[prepare] train records {train.height:,}  ({time.time()-t:.0f}s)")
    token_dict, comp_dict = build_dicts.build(train, gt)

    for split in args.splits.split(","):
        if split == "train":
            recs = train
        else:
            recs = read_sources(split).with_columns(
                pl.lit(None, dtype=pl.Utf8).alias("true_s1"), pl.lit(-1, dtype=pl.Int8).alias("bucket"))
            if args.dev_regex:  # dev smoke test only
                recs = recs.filter(pl.col("addr_raw").fill_null("").str.contains(args.dev_regex))
        recs = normalise(recs, token_dict, comp_dict)
        out = split_dir(split) / "records.parquet"
        recs.write_parquet(out)
        print(f"[prepare] {split}: {recs.height:,} records -> {out}  ({time.time()-t:.0f}s)")


if __name__ == "__main__":
    main()
