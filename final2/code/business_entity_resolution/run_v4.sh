#!/usr/bin/env bash
# v4 = v3 + cross-encoder feature.  Candidate set (HNSW + dense prefilter) is unchanged.
#   1. train the cross-encoder on the encoder-bucket candidate pairs of the test-like split
#   2. add its score `ce` to the train_dense and test features
#   3. retrain matcher + stage 2 on train_dense (tag "ce", prefilter pinned to "dense")
#      -> out-of-fold score on the test-like split (compare with v3: 0.98409)
#   4. predict the test set -> output_v4/
# Requires run_pipeline.sh, run_proxy.sh and run_final.sh (v3) to have run.
# Usage:  PY=/path/to/python bash run_v4.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error PYTHONUNBUFFERED=1
export TQDM_MININTERVAL=${TQDM_MININTERVAL:-5}
mkdir -p logs
{
  echo "================ cross-encoder train  ($(date '+%F %T'))"
  $PY src/cross_encoder.py train --split train_dense
  echo "================ cross-encoder score train_dense  ($(date '+%F %T'))"
  $PY src/cross_encoder.py score --split train_dense
  # GPU scores the test pairs while the CPU retrains the matcher (independent files)
  echo "================ cross-encoder score test (background, GPU)  ($(date '+%F %T'))"
  $PY src/cross_encoder.py score --split test > logs/v4_score_test.log 2>&1 &
  SCORE_PID=$!
  echo "================ matcher + stage 2 with ce  ($(date '+%F %T'))"
  ER_MODEL_TAG=ce ER_PREFILTER_TAG=dense $PY src/train_matcher.py --split train_dense
  ER_MODEL_TAG=ce ER_PREFILTER_TAG=dense $PY src/rescore.py --split train_dense
  echo "================ waiting for test scoring  ($(date '+%F %T'))"
  wait $SCORE_PID
  grep -a "\[ce\]" logs/v4_score_test.log
  echo "================ predict v4  ($(date '+%F %T'))"
  ER_MODEL_TAG=ce ER_PREFILTER_TAG=dense $PY src/predict.py --stage2 --out output_v4
  echo "================ done ($(date '+%F %T'))"
} 2>&1 | tee logs/v4.log
