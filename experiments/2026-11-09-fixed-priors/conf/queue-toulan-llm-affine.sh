#!/usr/bin/env bash
set -euo pipefail
while kill -0 902074 2>/dev/null; do sleep 20; done
if [[ ! -f /data/common/lgt/GroupSketch/experiments/2026-11-08-ccfe-full-ablation/logs/che-ccfe-none-exit-code.txt ]] || [[ $(cat /data/common/lgt/GroupSketch/experiments/2026-11-08-ccfe-full-ablation/logs/che-ccfe-none-exit-code.txt) != 0 ]]; then echo prerequisite_failed; exit 1; fi
while [[ -n $(nvidia-smi -i 3 --query-compute-apps=pid --format=csv,noheader) ]]; do sleep 15; done
exec bash /data/common/lgt/GroupSketch-worktrees/llm-affine-prior/experiments/2026-11-09-fixed-priors/conf/run-toulan-llm-affine.sh
