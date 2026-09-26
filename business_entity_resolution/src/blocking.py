"""Step 3 - candidate generation (blocking), stage 0: approximate nearest neighbours.

Every S2/S3 record belongs to at most one S1 entity, so we query FROM the S2/S3 side:
for each S2/S3 record we retrieve its top-K most similar S1 records of the same country
label by cosine similarity of the fine-tuned embeddings.  Country labels are an open
set: blocking simply iterates over whatever labels appear in the data.

Scalable search: a FAISS HNSW graph index (IndexHNSWFlat, inner product, M=32).  Each
query walks a navigable small-world graph over the S1 embeddings of its country and
evaluates only a few thousand candidates instead of all |S1| records; the search cost
grows ~ log |S1|, which is what makes the approach usable at billions of records.
(An inverted-file index was tried first: the fine-tuned embeddings are nearly isotropic
- mean cosine of random pairs 0.02 - so k-means lists separate them poorly and recall
dropped to ~95%; HNSW keeps 99.8% agreement with exact search.)  `--exact` switches to
brute-force GPU search for comparison.  The number of similarity evaluations actually
performed is taken from FAISS's own counters and reported against brute force.

Neighbours whose cosine is more than KNN_MAX_GAP below the record's best neighbour are
dropped.  The survivors go to the stage-1 prefilter (`prefilter.py`), whose output is
the final candidate set.

Output: WORK_DIR/<split>/candidates_knn.parquet with columns
    r_idx, s1_idx  : row numbers in records.parquet
    cos            : embedding cosine
    rank           : rank of this S1 among the K neighbours of the S2/S3 record (0 = best)
    cos1, cos2     : best and second-best neighbour cosine of the S2/S3 record

    python src/blocking.py --split train [--k 10] [--exact]
"""
import argparse
import time

import numpy as np
import polars as pl
from tqdm import tqdm

from config import (ENCODER_BUCKETS, HNSW_EF_CONSTRUCTION, HNSW_EF_SEARCH, HNSW_M, KNN_K, KNN_MAX_GAP, N_JOBS,
                    split_dir)

KNN_FILE = "candidates_knn.parquet"


def knn_exact(q: np.ndarray, d: np.ndarray, k: int, desc: str, chunk: int = 4096):
    """Brute-force top-k inner product on the GPU (reference implementation)."""
    import torch
    k = min(k, d.shape[0])
    dt, qt = torch.from_numpy(d).cuda(), torch.from_numpy(q).cuda()
    vals, idxs = [], []
    with torch.no_grad():
        for i in tqdm(range(0, qt.shape[0], chunk), desc=desc, unit="chunk"):
            v, ix = torch.topk(qt[i:i + chunk] @ dt.T, k, dim=1)
            vals.append(v.float().cpu())
            idxs.append(ix.cpu())
    del dt, qt
    torch.cuda.empty_cache()
    return torch.cat(vals).numpy(), torch.cat(idxs).numpy()


def knn_hnsw(q: np.ndarray, d: np.ndarray, k: int, desc: str, chunk: int = 200_000):
    """Approximate top-k inner product with a FAISS HNSW graph index (CPU, all cores).

    Returns (values, indices, number of distance computations actually performed)."""
    import faiss
    faiss.omp_set_num_threads(N_JOBS)
    d = np.ascontiguousarray(d, dtype=np.float32)
    index = faiss.IndexHNSWFlat(d.shape[1], HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.add(d)
    index.hnsw.efSearch = max(HNSW_EF_SEARCH, k)
    k = min(k, len(d))
    vals = np.empty((len(q), k), dtype=np.float32)
    idxs = np.empty((len(q), k), dtype=np.int64)
    faiss.cvar.hnsw_stats.reset()
    for i in tqdm(range(0, len(q), chunk), desc=f"{desc} (HNSW M={HNSW_M}, ef={index.hnsw.efSearch})", unit="chunk"):
        v, ix = index.search(np.ascontiguousarray(q[i:i + chunk], dtype=np.float32), k)
        vals[i:i + chunk], idxs[i:i + chunk] = v, ix
    return vals, idxs, float(faiss.cvar.hnsw_stats.ndis)


def run(split: str, k: int, max_gap: float = KNN_MAX_GAP, exact: bool = False):
    t = time.time()
    recs = pl.read_parquet(split_dir(split) / "records.parquet", columns=["src", "country"]).with_row_index("idx")
    emb = np.load(split_dir(split) / "emb.npy", mmap_mode="r")
    parts, n_compared, n_brute = [], 0, 0
    for country in sorted(recs["country"].unique().to_list()):
        c = recs.filter(pl.col("country") == country)
        s1_idx = c.filter(pl.col("src") == 1)["idx"].to_numpy()
        r_idx = c.filter(pl.col("src") != 1)["idx"].to_numpy()
        if len(s1_idx) == 0 or len(r_idx) == 0:
            continue
        d, q = np.ascontiguousarray(emb[s1_idx]), np.ascontiguousarray(emb[r_idx])
        desc = f"knn {split} {country}"
        if exact:
            vals, ix = knn_exact(q, d, k, desc)
            n_compared += len(q) * len(d)
        else:
            vals, ix, ndis = knn_hnsw(q, d, k, desc)
            n_compared += ndis
        n_brute += len(q) * len(d)
        kk = ix.shape[1]
        valid = ix >= 0  # an approximate index may return fewer than k hits
        vals = np.where(valid, vals, -np.inf).astype(np.float32)
        cos2 = vals[:, 1] if kk > 1 else np.full(len(r_idx), -1.0, dtype=np.float32)
        keep = valid & ((vals[:, :1] - vals) <= max_gap)
        rows = np.nonzero(keep)
        parts.append(pl.DataFrame({
            "r_idx": r_idx[rows[0]].astype(np.int32),
            "s1_idx": s1_idx[ix[rows]].astype(np.int32),
            "cos": vals[rows],
            "rank": rows[1].astype(np.int8),
            "cos1": vals[rows[0], 0],
            "cos2": np.where(np.isfinite(cos2), cos2, -1.0).astype(np.float32)[rows[0]],
        }))
        print(f"[blocking] {split} {country}: {len(r_idx):,} queries x {len(s1_idx):,} S1 "
              f"-> {parts[-1].height:,} pairs ({time.time()-t:.0f}s)", flush=True)
    cand = pl.concat(parts)
    cand.write_parquet(split_dir(split) / KNN_FILE)
    print(f"[blocking] {split}: {cand.height:,} pairs; similarity evaluations {n_compared:.3g} "
          f"vs brute force {n_brute:.3g} ({n_compared / n_brute:.2%}) ({time.time()-t:.0f}s)")
    return cand


def recall_report(split="train", ks=(1, 2, 3, 5, 10, 20)):
    """Pair recall of the kNN blocking for several K (train only)."""
    recs = pl.read_parquet(split_dir(split) / "records.parquet", columns=["entity_id", "true_s1", "bucket"])
    recs = recs.with_row_index("idx")
    cand = pl.read_parquet(split_dir(split) / KNN_FILE)
    s1_of = recs.select(pl.col("idx").alias("s1_idx"), pl.col("entity_id").alias("cand_s1"))
    # exclude the encoder's own training clusters: recall is measured out-of-sample
    r = recs.filter(pl.col("true_s1").is_not_null() & ~pl.col("bucket").is_in(list(ENCODER_BUCKETS))) \
        .select(pl.col("idx").alias("r_idx"), "true_s1", "bucket")
    hit = cand.join(s1_of, on="s1_idx").join(r, on="r_idx").filter(pl.col("cand_s1") == pl.col("true_s1"))
    n_true = r.height
    for k in ks:
        n = hit.filter(pl.col("rank") < k).select("r_idx").n_unique()
        print(f"  recall@{k:<3d} {n / n_true:.4f}   ({n:,}/{n_true:,} true pairs)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--k", type=int, default=KNN_K)
    ap.add_argument("--max-gap", type=float, default=KNN_MAX_GAP)
    ap.add_argument("--exact", action="store_true", help="brute-force GPU search instead of HNSW")
    args = ap.parse_args()
    run(args.split, args.k, args.max_gap, args.exact)
    if args.split == "train":
        recall_report("train", ks=[x for x in (1, 2, 3, 5, 10, 20, 30) if x <= args.k])


if __name__ == "__main__":
    main()
