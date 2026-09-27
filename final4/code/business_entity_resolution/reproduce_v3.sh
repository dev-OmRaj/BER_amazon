#!/usr/bin/env bash
# Data -> final (v3) outputs in one go:
#   run_pipeline.sh  (normalise, bi-encoder, HNSW blocking, prefilter, features, matcher, predict = v2)
#   run_proxy.sh     (test-like split, dense prefilter + matcher, stage-2 re-scorer)
#   run_final.sh     (v3 prediction -> output_v3/matching_results.tsv + candidate_pairs.tsv)
# Usage:  ER_DATA_DIR=/path/to/dataset PY=/path/to/python bash reproduce_v3.sh
set -euo pipefail
cd "$(dirname "$0")"
bash run_pipeline.sh
bash run_proxy.sh
bash run_final.sh
