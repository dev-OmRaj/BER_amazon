#!/usr/bin/env bash
# v6, part 2 (after run_v6_train.sh): score, retrain, predict.  Candidate set unchanged.
#   1. column ce_cf2 on train_dense: cfA2 and cfB2 score in PARALLEL (each only the fold it
#      never saw, see buckets.json), partial files merged -> same values as one process
#   2. test pairs: both models in parallel in the background (mean of both)
#   3. meanwhile retrain matcher + stage 2 with ce, ce_cf, ce_cf2 (tag "ce3", prefilter
#      pinned to "dense"; 16 OpenMP threads because the scorers also need CPU)
#      -> out-of-fold score on the test-like split (v6 gate: >= 0.98969, v5 = 0.98939)
#   4. merge test scores, predict -> output_v6/
# Usage:  PY=/path/to/python bash run_v6_rest.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error PYTHONUNBUFFERED=1
export TQDM_MININTERVAL=${TQDM_MININTERVAL:-5}
BATCH=${BATCH:-2048}
W=work
mkdir -p logs
{
  echo "================ score train_dense: cfA2 + cfB2 in parallel  ($(date '+%F %T'))"
  $PY src/cross_encoder.py score --split train_dense --models cfA2 --col ce_cf2 --batch $BATCH \
      --partial $W/train_dense/ce_cf2_part_A.parquet > logs/v6_score_train_A.log 2>&1 &
  PA=$!
  $PY src/cross_encoder.py score --split train_dense --models cfB2 --col ce_cf2 --batch $BATCH \
      --partial $W/train_dense/ce_cf2_part_B.parquet > logs/v6_score_train_B.log 2>&1 &
  PB=$!
  wait $PA
  wait $PB
  $PY src/cross_encoder.py merge --split train_dense --col ce_cf2 \
      --partials $W/train_dense/ce_cf2_part_A.parquet,$W/train_dense/ce_cf2_part_B.parquet

  echo "================ score test: cfA2 + cfB2 in parallel (background)  ($(date '+%F %T'))"
  $PY src/cross_encoder.py score --split test --models cfA2 --col ce_cf2 --batch $BATCH \
      --partial $W/test/ce_cf2_part_A.parquet > logs/v6_score_test_A.log 2>&1 &
  TA=$!
  $PY src/cross_encoder.py score --split test --models cfB2 --col ce_cf2 --batch $BATCH \
      --partial $W/test/ce_cf2_part_B.parquet > logs/v6_score_test_B.log 2>&1 &
  TB=$!

  echo "================ matcher + stage 2 with ce + ce_cf + ce_cf2  ($(date '+%F %T'))"
  ER_OMP_THREADS=16 ER_MODEL_TAG=ce3 ER_PREFILTER_TAG=dense $PY src/train_matcher.py --split train_dense
  ER_OMP_THREADS=16 ER_MODEL_TAG=ce3 ER_PREFILTER_TAG=dense $PY src/rescore.py --split train_dense

  echo "================ waiting for test scoring  ($(date '+%F %T'))"
  wait $TA
  wait $TB
  grep -a "\[ce\]" logs/v6_score_test_A.log logs/v6_score_test_B.log
  $PY src/cross_encoder.py merge --split test --col ce_cf2 \
      --partials $W/test/ce_cf2_part_A.parquet,$W/test/ce_cf2_part_B.parquet

  echo "================ predict v6  ($(date '+%F %T'))"
  ER_MODEL_TAG=ce3 ER_PREFILTER_TAG=dense $PY src/predict.py --stage2 --out output_v6
  echo "================ done ($(date '+%F %T'))"
} 2>&1 | tee logs/v6_rest.log
