#!/usr/bin/env bash
# End-to-end pipeline: raw TSVs -> normalisation -> bi-encoder -> ANN blocking -> prefilter -> features
# -> LightGBM matcher -> output/matching_results.tsv + output/candidate_pairs.tsv
#
# Usage:  PY=/path/to/python bash run_pipeline.sh            (all steps)
#         PY=/path/to/python bash run_pipeline.sh features    (resume from a step)
# Steps:  prepare encoder embed blocking prefilter features matcher predict
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error PYTHONUNBUFFERED=1
# tqdm progress bars: refresh at most every 5 s so the log files stay readable
export TQDM_MININTERVAL=${TQDM_MININTERVAL:-5}
mkdir -p logs
START=${1:-prepare}
STEPS=(prepare encoder embed blocking prefilter features matcher predict)

run=0
for s in "${STEPS[@]}"; do
  [[ "$s" == "$START" ]] && run=1
  [[ $run == 0 ]] && continue
  echo "================ $s  ($(date '+%F %T'))"
  case $s in
    prepare)  $PY src/prepare.py 2>&1 | tee logs/prepare.log ;;
    encoder)  $PY src/encoder.py train 2>&1 | tee logs/encoder.log ;;
    embed)    { $PY src/encoder.py embed --split train && $PY src/encoder.py embed --split test; } 2>&1 | tee logs/embed.log ;;
    blocking) { $PY src/blocking.py --split train && $PY src/blocking.py --split test; } 2>&1 | tee logs/blocking.log ;;
    prefilter) { $PY src/prefilter.py --split train && $PY src/prefilter.py --split test; } 2>&1 | tee logs/prefilter.log ;;
    features) { $PY src/features.py --split train && $PY src/features.py --split test; } 2>&1 | tee logs/features.log ;;
    matcher)  $PY src/train_matcher.py --transfer 2>&1 | tee logs/matcher.log ;;
    predict)  $PY src/predict.py 2>&1 | tee logs/predict.log ;;
  esac
done
echo "================ done ($(date '+%F %T'))"
