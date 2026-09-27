#!/usr/bin/env bash
# Offline validation on a test-like training split (no leaderboard upload needed).
#
#   phase "check"  : build train_dense (same distractor density as test), score it with the
#                    SAVED v2 models out-of-fold -> should reproduce the v2 leaderboard (0.9775)
#   phase "dense"  : retrain prefilter + matcher on train_dense (models tagged "dense",
#                    v2 models untouched) -> out-of-fold score on the same test-like split
#   phase "stage2" : train + evaluate the stage-2 cluster-context re-scorer on top of the
#                    dense matcher (out-of-fold, same split)
#
# Usage:  PY=/path/to/python bash run_proxy.sh            (all phases)
#         PY=/path/to/python bash run_proxy.sh dense      (from the dense phase)
#         PY=/path/to/python bash run_proxy.sh stage2     (stage-2 phase only)
# Requires the v2 pipeline to have run (work/train, work/test, work/prefilter, work/matcher).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error PYTHONUNBUFFERED=1
export TQDM_MININTERVAL=${TQDM_MININTERVAL:-5}
mkdir -p logs
START=${1:-check}

if [[ "$START" == "check" ]]; then
  echo "================ check: densify + v2 models on the test-like split  ($(date '+%F %T'))"
  {
    $PY src/densify.py
    $PY src/blocking.py --split train_dense
    $PY src/prefilter.py --split train_dense --reuse
    $PY src/features.py --split train_dense
    $PY src/train_matcher.py --split train_dense --reuse
  } 2>&1 | tee logs/proxy_check.log
fi

if [[ "$START" == "check" || "$START" == "dense" ]]; then
  echo "================ dense: retrain prefilter + matcher on the test-like split  ($(date '+%F %T'))"
  {
    ER_MODEL_TAG=dense $PY src/prefilter.py --split train_dense
    $PY src/features.py --split train_dense
    ER_MODEL_TAG=dense $PY src/train_matcher.py --split train_dense
  } 2>&1 | tee logs/proxy_dense.log
fi

echo "================ stage2: cluster-context re-scorer on the dense matcher  ($(date '+%F %T'))"
ER_MODEL_TAG=dense $PY src/rescore.py --split train_dense 2>&1 | tee logs/proxy_stage2.log
echo "================ done ($(date '+%F %T'))"
