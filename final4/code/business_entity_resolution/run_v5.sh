#!/usr/bin/env bash
# v5 = v4 + cross-fitted cross-encoders.  Candidate set (HNSW + dense prefilter) unchanged.
#   1. train two more cross-encoders, starting from the v4 one: cfA on fold-A candidate
#      pairs, cfB on fold-B pairs (in parallel on the GPU)
#   2. column ce_cf: fold-A pairs scored by cfB, fold-B pairs by cfA (out-of-fold);
#      test pairs by the mean of both
#   3. retrain matcher + stage 2 with ce and ce_cf (tag "ce2", prefilter pinned to "dense")
#      -> out-of-fold score on the test-like split (compare with v4: 0.98863)
#   4. predict the test set -> output_v5/
# Requires run_v4.sh (v4) to have run.
# Usage:  PY=/path/to/python bash run_v5.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error PYTHONUNBUFFERED=1
export TQDM_MININTERVAL=${TQDM_MININTERVAL:-5}
MAX_PAIRS=${MAX_PAIRS:-1500000}
mkdir -p logs
{
  echo "================ train cross-encoders cfA + cfB (parallel)  ($(date '+%F %T'))"
  $PY src/cross_encoder.py train --buckets A --init ce --out cfA --max-pairs "$MAX_PAIRS" > logs/v5_train_cfA.log 2>&1 &
  PID_A=$!
  $PY src/cross_encoder.py train --buckets B --init ce --out cfB --max-pairs "$MAX_PAIRS" > logs/v5_train_cfB.log 2>&1 &
  PID_B=$!
  wait $PID_A
  wait $PID_B
  grep -a "\[ce\]" logs/v5_train_cfA.log logs/v5_train_cfB.log
  echo "================ score train_dense (cross-fitted ce_cf)  ($(date '+%F %T'))"
  $PY src/cross_encoder.py score --split train_dense --models cfA,cfB --col ce_cf
  # GPU scores the test pairs while the CPU retrains the matcher (independent files)
  echo "================ score test (background, GPU)  ($(date '+%F %T'))"
  $PY src/cross_encoder.py score --split test --models cfA,cfB --col ce_cf > logs/v5_score_test.log 2>&1 &
  SCORE_PID=$!
  echo "================ matcher + stage 2 with ce + ce_cf  ($(date '+%F %T'))"
  ER_MODEL_TAG=ce2 ER_PREFILTER_TAG=dense $PY src/train_matcher.py --split train_dense
  ER_MODEL_TAG=ce2 ER_PREFILTER_TAG=dense $PY src/rescore.py --split train_dense
  echo "================ waiting for test scoring  ($(date '+%F %T'))"
  wait $SCORE_PID
  grep -a "\[ce\]" logs/v5_score_test.log
  echo "================ predict v5  ($(date '+%F %T'))"
  ER_MODEL_TAG=ce2 ER_PREFILTER_TAG=dense $PY src/predict.py --stage2 --out output_v5
  echo "================ done ($(date '+%F %T'))"
} 2>&1 | tee logs/v5.log
