#!/usr/bin/env bash
set -uo pipefail
cd /data/common/lgt/GroupSketch-worktrees/llm-affine-prior
source tools/env.sh
export CUDA_VISIBLE_DEVICES=3
python -u animate_svg.py --target /data/common/lgt/GroupSketch-worktrees/llm-affine-prior/experiments/2026-11-09-llm-affine-prior/inputs/toulan-affine-v2/input --caption 'A person shoots a basketball toward the hoop.' --num_frames 24 --num_iter 1000 --seed 1000 --dino --timeE --affine_prior /data/common/lgt/GroupSketch-worktrees/llm-affine-prior/experiments/2026-11-09-llm-affine-prior/inputs/toulan-affine-v2/input/affine-prior.yaml --output_folder 2026-11-09-fixed-priors/toulan-llm-affine --progress_callback_file /data/common/lgt/GroupSketch-worktrees/llm-affine-prior/experiments/2026-11-09-fixed-priors/logs/toulan-llm-affine-progress.json
rc=$?
printf "%s\n" "$rc" > /data/common/lgt/GroupSketch-worktrees/llm-affine-prior/experiments/2026-11-09-fixed-priors/logs/toulan-llm-affine-exit-code.txt
exit "$rc"
