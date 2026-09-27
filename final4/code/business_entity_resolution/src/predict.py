"""Step 6 - score the test candidates and write the two submission files.

    python src/predict.py

Writes OUTPUT_DIR/matching_results.tsv and OUTPUT_DIR/candidate_pairs.tsv
(one row per test Source-1 entity, empty list when nothing matched / no candidate)
and runs the official validator if it can be found.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from tqdm import tqdm

from config import DATA_DIR, OMP_THREADS, OUTPUT_DIR, split_dir
from decision import decide
from io_utils import write_id_lists
from train_matcher import MATCHER_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=str(OUTPUT_DIR))
    ap.add_argument("--stage2", action="store_true", help="re-score with the stage-2 cluster-context model")
    args = ap.parse_args()
    t = time.time()

    report = json.loads((MATCHER_DIR / "report.json").read_text())
    models = [lgb.Booster(model_file=str(MATCHER_DIR / f"model_{k}.txt")) for k in ("A", "B")]
    cols = models[0].feature_name()

    d = split_dir(args.split)
    recs = pl.read_parquet(d / "records.parquet", columns=["entity_id", "src"])
    ids = recs["entity_id"].to_numpy()
    feats = pl.read_parquet(d / "features.parquet")
    x = feats.select(cols).to_numpy()
    p = np.mean([m.predict(x, num_threads=OMP_THREADS) for m in tqdm(models, desc="score test pairs", unit="model")], axis=0)
    del x
    pairs = pl.DataFrame({
        "s1_id": ids[feats["s1_idx"].to_numpy()],
        "r_id": ids[feats["r_idx"].to_numpy()],
        "p": p, "cos": feats["cos"].to_numpy(),
    })
    print(f"[predict] scored {pairs.height:,} candidate pairs ({time.time()-t:.0f}s)")
    if args.stage2:
        import rescore
        del feats
        pairs = rescore.apply(args.split, p)
        report = json.loads((rescore.STAGE2_DIR / "report.json").read_text())["stage2"]
        print(f"[predict] stage-2 re-scoring done ({time.time()-t:.0f}s)")

    matches = decide(pairs, report["rule"], report["threshold"])
    s1_ids = recs.filter(pl.col("src") == 1)["entity_id"]
    out = Path(args.out)
    write_id_lists(out / "candidate_pairs.tsv", s1_ids, pairs.select("s1_id", "r_id"), "candidate_entity_ids")
    res = write_id_lists(out / "matching_results.tsv", s1_ids, matches, "matched_entity_ids")
    n_empty = (res["matched_entity_ids"] == "").sum()
    print(f"[predict] {matches.height:,} matches for {s1_ids.len():,} S1 entities "
          f"({n_empty:,} predicted singletons) using rule={report['rule']} t={report['threshold']}")
    print(f"[predict] wrote {out/'matching_results.tsv'} and {out/'candidate_pairs.tsv'} ({time.time()-t:.0f}s)")

    validator = DATA_DIR.parent / "utils" / "validate_submission.py"
    if args.split.startswith("test") and validator.exists():
        subprocess.run([sys.executable, str(validator), "--matching", str(out / "matching_results.tsv"),
                        "--candidate", str(out / "candidate_pairs.tsv"), "--test-dir", str(DATA_DIR / "test")])


if __name__ == "__main__":
    main()
