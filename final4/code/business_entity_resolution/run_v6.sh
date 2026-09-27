#!/usr/bin/env bash
# v6 = v5 + a second round of cross-fitted cross-encoders (candidate set unchanged):
#   run_v6_train.sh  continue cfA / cfB on 1.5M new pairs of their fold -> cfA2 / cfB2
#   run_v6_rest.sh   parallel out-of-fold scoring (ce_cf2), matcher + stage 2 (tag ce3),
#                    predict -> output_v6/
# Requires run_v5.sh (v5) to have run.
# Usage:  PY=/path/to/python bash run_v6.sh
set -euo pipefail
cd "$(dirname "$0")"
bash run_v6_train.sh
bash run_v6_rest.sh
