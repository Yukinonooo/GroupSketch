#!/usr/bin/env bash
set -euo pipefail
while kill -0 3530170 2>/dev/null; do sleep 20; done
if [[ ! -f /data/common/lgt/GroupSketch-worktrees/llm-affine-prior/experiments/2026-11-09-fixed-priors/logs/toulan-human-exit-code.txt ]] || [[ $(cat /data/common/lgt/GroupSketch-worktrees/llm-affine-prior/experiments/2026-11-09-fixed-priors/logs/toulan-human-exit-code.txt) != 0 ]]; then echo prerequisite_failed; exit 1; fi
while [[ -n $(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader) ]]; do sleep 15; done
exec bash /data/common/lgt/GroupSketch-worktrees/llm-affine-prior/experiments/2026-11-09-fixed-priors/conf/run-che-llm-affine.sh
