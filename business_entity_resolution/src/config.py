"""Paths and global settings for the entity-resolution pipeline.

Every path can be overridden with an environment variable so the pipeline can be
run from any checkout location:

    ER_DATA_DIR   folder that contains train/ and test/ (the challenge `dataset/` dir)
    ER_WORK_DIR   folder for intermediate artefacts (parquet, embeddings, models)
    ER_OUTPUT_DIR folder where matching_results.tsv / candidate_pairs.tsv are written
"""
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.environ.get(
    "ER_DATA_DIR",
    PROJECT_DIR.parent / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset",
))
WORK_DIR = Path(os.environ.get("ER_WORK_DIR", PROJECT_DIR / "work"))
OUTPUT_DIR = Path(os.environ.get("ER_OUTPUT_DIR", PROJECT_DIR / "output"))

SEED = 42
N_JOBS = int(os.environ.get("ER_N_JOBS", os.cpu_count() or 8))

# ---------------------------------------------------------------- folds (train only)
# Each Source-1 cluster (the S1 record + all its true S2/S3 matches) is hashed into
# one of 10 buckets; unmatched S2/S3 records are hashed by their own id.
#   buckets 0-1 : used to learn the normalisation dictionaries and to fine-tune the
#                 bi-encoder (never used to train/evaluate the matcher)
#   buckets 2-5 : matcher fold A
#   buckets 6-9 : matcher fold B   (A and B are cross-fitted -> out-of-fold scores)
N_BUCKETS = 10
ENCODER_BUCKETS = (0, 1)
FOLD_A_BUCKETS = (2, 3, 4, 5)
FOLD_B_BUCKETS = (6, 7, 8, 9)

# ---------------------------------------------------------------- bi-encoder
ENCODER_BASE_MODEL = os.environ.get("ER_ENCODER_BASE", "intfloat/multilingual-e5-small")  # MIT, 118M params
ENCODER_MAX_LEN = 64
ENCODER_DIR = WORK_DIR / "encoder"

# ---------------------------------------------------------------- blocking
KNN_K = int(os.environ.get("ER_KNN_K", 10))          # S1 neighbours retrieved per S2/S3 record
# keep a neighbour only if its cosine is within KNN_MAX_GAP of the record's best neighbour
# (dev slice: gap<=0.2 keeps the same recall as all 10 neighbours with 6x fewer pairs)
KNN_MAX_GAP = float(os.environ.get("ER_KNN_MAX_GAP", 0.25))
# HNSW graph index (FAISS): links per node, build / search beam widths
HNSW_M = int(os.environ.get("ER_HNSW_M", 32))
HNSW_EF_CONSTRUCTION = int(os.environ.get("ER_HNSW_EF_CONSTRUCTION", 200))
HNSW_EF_SEARCH = int(os.environ.get("ER_HNSW_EF_SEARCH", 128))

# ---------------------------------------------------------------- stage-1 prefilter
# the cheap stage-1 model keeps the smallest candidate set that still contains this
# share of the true pairs found by the kNN stage (chosen on out-of-fold scores)
PREFILTER_RECALL = float(os.environ.get("ER_PREFILTER_RECALL", 0.9995))


def pool():
    """Process pool using fork: workers inherit module globals (the normaliser, the
    hashing vectoriser).  Python 3.14 changed the Linux default to forkserver, so the
    start method is requested explicitly."""
    import multiprocessing
    return multiprocessing.get_context("fork").Pool(N_JOBS)


def pmap(fn, items, desc, chunksize=1000):
    """Order-preserving parallel map over a list, with a tqdm progress bar."""
    from tqdm import tqdm
    with pool() as p:
        return list(tqdm(p.imap(fn, items, chunksize=chunksize), total=len(items), desc=desc,
                         unit="it", unit_scale=True, smoothing=0.05))


def split_dir(split: str) -> Path:
    d = WORK_DIR / split
    d.mkdir(parents=True, exist_ok=True)
    return d
