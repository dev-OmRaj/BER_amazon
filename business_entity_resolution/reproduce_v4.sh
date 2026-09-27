#!/usr/bin/env bash
# Data -> final (v4) outputs in one go:
#   run_pipeline.sh  (normalise, bi-encoder, HNSW blocking, prefilter, features, matcher, predict = v2)
#   run_proxy.sh     (test-like split, dense prefilter + matcher, stage-2 re-scorer)
#   run_final.sh     (v3 prediction: dense prefilter -> test candidates + features)
#   run_v4.sh        (cross-encoder feature, matcher + stage 2 retrained -> output_v4/)
# Usage:  ER_DATA_DIR=/path/to/dataset PY=/path/to/python bash reproduce_v4.sh
set -euo pipefail
cd "$(dirname "$0")"
bash run_pipeline.sh
bash run_proxy.sh
bash run_final.sh
bash run_v4.sh
