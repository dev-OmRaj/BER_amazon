#!/usr/bin/env bash
# v6, part 1: continue cfA / cfB on 1.5M NEW pairs of their own fold (pairs used by v5 are
# excluded), lr 2e-5, both in parallel on the GPU -> work/cross_encoder_cfA2, _cfB2.
# Same recipe as v5 (batch 128).  Part 2 is run_v6_rest.sh.
# Usage:  PY=/path/to/python bash run_v6_train.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error PYTHONUNBUFFERED=1
export TQDM_MININTERVAL=${TQDM_MININTERVAL:-5}
MAX_PAIRS=${MAX_PAIRS:-1500000}
PREV_PAIRS=${PREV_PAIRS:-1500000}   # --max-pairs used by run_v5.sh
mkdir -p logs
echo "================ train cross-encoders cfA2 + cfB2 (parallel)  ($(date '+%F %T'))"
$PY src/cross_encoder.py train --buckets A --init cfA --out cfA2 --max-pairs "$MAX_PAIRS" \
    --exclude-prev "$PREV_PAIRS" --lr 2e-5 > logs/v6_train_cfA2.log 2>&1 &
PID_A=$!
$PY src/cross_encoder.py train --buckets B --init cfB --out cfB2 --max-pairs "$MAX_PAIRS" \
    --exclude-prev "$PREV_PAIRS" --lr 2e-5 > logs/v6_train_cfB2.log 2>&1 &
PID_B=$!
wait $PID_A
wait $PID_B
grep -a "\[ce\]" logs/v6_train_cfA2.log logs/v6_train_cfB2.log
echo "================ training done ($(date '+%F %T'))"
