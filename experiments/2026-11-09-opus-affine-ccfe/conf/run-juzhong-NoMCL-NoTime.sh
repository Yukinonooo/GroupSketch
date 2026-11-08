#!/usr/bin/env bash
set -uo pipefail
cd /data/common/lgt/GroupSketch-worktrees/llm-affine-prior
source tools/env.sh
export CUDA_VISIBLE_DEVICES=7
python -u animate_svg.py --target /data/common/lgt/GroupSketch-worktrees/llm-affine-prior/experiments/2026-11-09-llm-affine-prior/inputs/juzhong-affine-v2/input --caption 'A weightlifter performs a squat-style lift, moving the barbell up and down while holding it with both hands.' --num_frames 16 --num_iter 1000 --seed 1000 --affine_prior /data/common/lgt/GroupSketch-worktrees/llm-affine-prior/experiments/2026-11-09-llm-affine-prior/inputs/juzhong-affine-v2/input/affine-prior.yaml --output_folder 2026-11-09-opus-affine-ccfe/juzhong-NoMCL-NoTime --progress_callback_file /data/common/lgt/GroupSketch-worktrees/llm-affine-prior/experiments/2026-11-09-opus-affine-ccfe/logs/juzhong-NoMCL-NoTime-progress.json
rc=$?
printf "%s\n" "$rc" > /data/common/lgt/GroupSketch-worktrees/llm-affine-prior/experiments/2026-11-09-opus-affine-ccfe/logs/juzhong-NoMCL-NoTime-exit-code.txt
exit "$rc"
