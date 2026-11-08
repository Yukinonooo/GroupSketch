#!/usr/bin/env bash
set -uo pipefail
cd /data/common/lgt/GroupSketch-worktrees/llm-affine-prior
source tools/env.sh
export CUDA_VISIBLE_DEVICES=2
python -u animate_svg.py --target /data/common/lgt/GroupSketch-worktrees/llm-affine-prior/experiments/2026-11-09-llm-affine-prior/inputs/che-affine-v2/input --caption 'A car moves forward across the zebra crossing while the traffic light pole stays still.' --num_frames 16 --num_iter 1000 --seed 1000 --timeE --affine_prior /data/common/lgt/GroupSketch-worktrees/llm-affine-prior/experiments/2026-11-09-llm-affine-prior/inputs/che-affine-v2/input/affine-prior.yaml --output_folder 2026-11-09-opus-affine-ccfe/che-NoMCL --progress_callback_file /data/common/lgt/GroupSketch-worktrees/llm-affine-prior/experiments/2026-11-09-opus-affine-ccfe/logs/che-NoMCL-progress.json
rc=$?
printf "%s\n" "$rc" > /data/common/lgt/GroupSketch-worktrees/llm-affine-prior/experiments/2026-11-09-opus-affine-ccfe/logs/che-NoMCL-exit-code.txt
exit "$rc"
