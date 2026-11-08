#!/usr/bin/env bash
set -uo pipefail
cd /data/common/lgt/GroupSketch
source tools/env.sh
export CUDA_VISIBLE_DEVICES=1
python -u animate_svg.py --target experiments/2026-11-08-server-bringup/inputs/toulan/input --caption 'A person shoots a basketball toward the hoop.' --num_iter 1000 --num_frames 24 --seed 1000 --output_folder 2026-11-08-ccfe-full-ablation/ccfe-none --progress_callback_file /data/common/lgt/GroupSketch/experiments/2026-11-08-ccfe-full-ablation/logs/ccfe-none-progress.json
rc=$?
printf "%s\n" "$rc" > /data/common/lgt/GroupSketch/experiments/2026-11-08-ccfe-full-ablation/logs/ccfe-none-exit-code.txt
exit "$rc"
