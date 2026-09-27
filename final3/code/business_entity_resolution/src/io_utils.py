"""Reading the challenge TSVs and writing the submission TSVs."""
import zlib
from pathlib import Path

import polars as pl
from tqdm import tqdm

from config import DATA_DIR, N_BUCKETS


def read_tsv(path: Path) -> pl.DataFrame:
    # quote_char=None: names contain stray quotes; every column is read as string
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema_length=0)


def read_sources(split: str) -> pl.DataFrame:
    """All three sources of a split stacked, with integer `src` column (1/2/3)."""
    frames = []
    for s in (1, 2, 3):
        df = read_tsv(DATA_DIR / split / f"{split}_source{s}.tsv")
        frames.append(df.with_columns(pl.lit(s, dtype=pl.Int8).alias("src")))
    return pl.concat(frames).rename({"business_name": "name_raw", "business_address": "addr_raw"})


def read_ground_truth() -> pl.DataFrame:
    """Long format: one row per (s1_id, r_id) true pair."""
    gt = read_tsv(DATA_DIR / "train" / "train_ground_truth.tsv")
    return (
        gt.with_columns(pl.col("matched_entity_ids").str.split(","))
        .explode("matched_entity_ids")
        .filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
        .rename({"source1_entity_id": "s1_id", "matched_entity_ids": "r_id"})
    )


def bucket_of(ids: pl.Series) -> pl.Series:
    """Stable (library-version independent) hash bucket of an id."""
    return pl.Series([zlib.crc32(x.encode()) % N_BUCKETS for x in ids.to_list()], dtype=pl.Int8)


def write_id_lists(path: Path, s1_ids: pl.Series, pairs: pl.DataFrame, col_name: str):
    """Write `source1_entity_id \t <col_name>` with one row per S1 id.

    pairs: DataFrame with columns s1_id, r_id (only S2/S3 ids).
    """
    grouped = pairs.unique(["s1_id", "r_id"]).sort(["s1_id", "r_id"]).group_by("s1_id").agg(
        pl.col("r_id").str.join(",").alias(col_name))
    out = (
        pl.DataFrame({"source1_entity_id": s1_ids})
        .join(grouped, left_on="source1_entity_id", right_on="s1_id", how="left")
        .with_columns(pl.col(col_name).fill_null(""))
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    # write manually: polars would quote nothing, but we want an exact, simple TSV
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{col_name}\n")
        for a, b in tqdm(out.iter_rows(), total=out.height, desc=f"write {path.name}", unit="row",
                         unit_scale=True):
            f.write(f"{a}\t{b}\n")
    return out
