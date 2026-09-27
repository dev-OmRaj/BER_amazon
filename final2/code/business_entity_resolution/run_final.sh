#!/usr/bin/env bash
# v3 test-set prediction: dense-trained prefilter + matcher + stage-2 re-scorer.
# Reuses the v2 test artefacts (records, embeddings, HNSW kNN candidates) and the models
# trained by run_proxy.sh.  Writes to output_v3/ so output/ (v2) stays untouched.
#
# Usage:  PY=/path/to/python bash run_final.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error PYTHONUNBUFFERED=1
export TQDM_MININTERVAL=${TQDM_MININTERVAL:-5}
export ER_MODEL_TAG=dense
mkdir -p logs
echo "================ final (v3): prefilter + features + matcher + stage 2 on test  ($(date '+%F %T'))"
{
  $PY src/prefilter.py --split test
  $PY src/features.py --split test
  $PY src/predict.py --stage2 --out output_v3
} 2>&1 | tee logs/final_v3.log
echo "================ done ($(date '+%F %T'))"
