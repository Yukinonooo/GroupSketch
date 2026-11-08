#!/usr/bin/env bash
set -uo pipefail
cd /data/common/lgt/GroupSketch-worktrees/llm-affine-prior
source tools/env.sh
export CUDA_VISIBLE_DEVICES=4
python -u animate_svg.py --target /data/common/lgt/GroupSketch/experiments/2026-11-08-server-bringup-r1/inputs/dancers/input --caption 'Two people turn a long jump rope while two people in the middle jump over it.' --num_frames 16 --num_iter 1000 --seed 1000 --dino --timeE --output_folder 2026-11-09-fixed-priors/dancers-human --progress_callback_file /data/common/lgt/GroupSketch-worktrees/llm-affine-prior/experiments/2026-11-09-fixed-priors/logs/dancers-human-progress.json
rc=$?
printf "%s\n" "$rc" > /data/common/lgt/GroupSketch-worktrees/llm-affine-prior/experiments/2026-11-09-fixed-priors/logs/dancers-human-exit-code.txt
exit "$rc"
