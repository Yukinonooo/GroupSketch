#!/usr/bin/env bash
set -euo pipefail
while kill -0 2306728 2>/dev/null; do sleep 30; done
if [[ ! -f /data/common/lgt/GroupSketch/experiments/2026-11-08-server-bringup-r1/logs/juzhong-ccfe-off-exit-code.txt ]] || [[ $(cat /data/common/lgt/GroupSketch/experiments/2026-11-08-server-bringup-r1/logs/juzhong-ccfe-off-exit-code.txt) != 0 ]]; then echo prerequisite_failed; exit 1; fi
while [[ -n $(nvidia-smi -i 7 --query-compute-apps=pid --format=csv,noheader) ]]; do sleep 15; done
exec bash /data/common/lgt/GroupSketch/experiments/2026-11-08-ccfe-full-ablation/conf/run-juzhong-ccfe-none.sh
