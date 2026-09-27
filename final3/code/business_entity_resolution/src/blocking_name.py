"""Blocking channel 2 (v6): name-only search for S2/S3 records WITHOUT an address.

Error analysis (test-like split): 1.08% of true pairs never become candidates; 70% of those
belong to records without an address.  Their embedding query is "name | <empty>", which is
compared with S1 texts that do have an address, so the true S1 often falls outside the
top-K.  This channel embeds the S1 records by NAME ONLY (same text format as the query:
"name | ") with the same bi-encoder, builds one HNSW index per country label, and lets
each no-address record retrieve its top-K S1 by name.  New pairs are merged into the
stage-0 candidate set; `cos` stays the usual full-record cosine, rank / cos1 / cos2 are
recomputed over the merged set and `chan` = 1 marks pairs found only by this channel.

The result is written to a NEW split folder (records and embeddings are linked, not
copied), so the v5 artefacts are never touched:

    python src/blocking_name.py --src-split train_dense --split train_dense6
    python src/blocking_name.py --src-split test --split test6
"""
import argparse
import os
import time

import numpy as np
import polars as pl
import torch
from tqdm import tqdm

from blocking import KNN_FILE, knn_hnsw
from config import ENCODER_DIR, split_dir
from encoder import Encoder

NAME_K = 5


@torch.no_grad()
def name_embeddings(names: list, batch=2048) -> np.ndarray:
    enc = Encoder(str(ENCODER_DIR))
    enc.model.eval().half()
    texts = [f"query: {n} | " for n in names]
    order = np.argsort([len(x) for x in texts])
    out = np.empty((len(texts), enc.model.config.hidden_size), dtype=np.float16)
    for i in tqdm(range(0, len(texts), batch), desc="name-only S1 embeddings", unit="batch"):
        idx = order[i:i + batch]
        out[idx] = enc.encode_batch([texts[j] for j in idx]).cpu().numpy().astype(np.float16)
    return out


def link(src: str, dst: str, name: str):
    target = split_dir(src) / name
    link_path = split_dir(dst) / name
    if not link_path.exists():
        os.symlink(target.resolve(), link_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-split", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--k", type=int, default=NAME_K)
    args = ap.parse_args()
    t = time.time()
    for f in ("records.parquet", "emb.npy"):
        link(args.src_split, args.split, f)
    d = split_dir(args.split)
    recs = pl.read_parquet(d / "records.parquet", columns=["src", "country", "name", "addr_missing"]) \
        .with_row_index("idx")
    emb = np.load(d / "emb.npy", mmap_mode="r")
    old = pl.read_parquet(split_dir(args.src_split) / KNN_FILE, columns=["r_idx", "s1_idx", "cos"])
    s1_all = recs.filter(pl.col("src") == 1)
    name_emb = name_embeddings(s1_all["name"].to_list())
    pos = {c: i for i, c in enumerate(s1_all["idx"].to_list())}
    parts = []
    for country in sorted(recs["country"].unique().to_list()):
        s1 = s1_all.filter(pl.col("country") == country)["idx"].to_numpy()
        q = recs.filter((pl.col("country") == country) & (pl.col("src") != 1) & pl.col("addr_missing"))["idx"] \
            .to_numpy()
        if len(s1) == 0 or len(q) == 0:
            continue
        d_emb = name_emb[[pos[i] for i in s1]]
        _, ix, _ = knn_hnsw(np.ascontiguousarray(emb[q]), d_emb, args.k, f"name channel {args.split} {country}")
        rows = np.nonzero(ix >= 0)
        parts.append(pl.DataFrame({"r_idx": q[rows[0]].astype(np.int32),
                                   "s1_idx": s1[ix[rows]].astype(np.int32)}))
        print(f"[name-channel] {country}: {len(q):,} no-address records x {len(s1):,} S1 "
              f"-> {parts[-1].height:,} pairs", flush=True)
    new = pl.concat(parts).join(old.select("r_idx", "s1_idx"), on=["r_idx", "s1_idx"], how="anti")
    a, b = new["r_idx"].to_numpy(), new["s1_idx"].to_numpy()
    cos = np.einsum("ij,ij->i", emb[a].astype(np.float32), emb[b].astype(np.float32)).astype(np.float32)
    new = new.with_columns(pl.Series("cos", cos), pl.lit(1, dtype=pl.Int8).alias("chan"))
    merged = pl.concat([old.with_columns(pl.lit(0, dtype=pl.Int8).alias("chan")), new])
    merged = merged.sort(["r_idx", "cos"], descending=[False, True]).with_columns(
        (pl.int_range(pl.len()).over("r_idx")).cast(pl.Int8).alias("rank"),
        pl.col("cos").max().over("r_idx").alias("cos1"))
    second = merged.filter(pl.col("rank") == 1).select("r_idx", pl.col("cos").alias("cos2"))
    merged = merged.join(second, on="r_idx", how="left").with_columns(pl.col("cos2").fill_null(-1.0)) \
        .select("r_idx", "s1_idx", "cos", "rank", "cos1", "cos2", "chan")
    merged.write_parquet(d / KNN_FILE)
    print(f"[name-channel] {args.split}: {new.height:,} new pairs added to {old.height:,} "
          f"-> {merged.height:,} ({time.time()-t:.0f}s)")


if __name__ == "__main__":
    main()
