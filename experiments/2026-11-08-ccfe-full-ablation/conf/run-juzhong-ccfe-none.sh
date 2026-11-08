#!/usr/bin/env bash
set -uo pipefail
cd /data/common/lgt/GroupSketch
source tools/env.sh
export CUDA_VISIBLE_DEVICES=7
python -u animate_svg.py --target experiments/2026-11-08-server-bringup-r1/inputs/juzhong/input --caption 'A weightlifter performs a squat-style lift, moving the barbell up and down while holding it with both hands.' --num_iter 1000 --num_frames 16 --seed 1000 --output_folder 2026-11-08-ccfe-full-ablation/juzhong-ccfe-none --progress_callback_file /data/common/lgt/GroupSketch/experiments/2026-11-08-ccfe-full-ablation/logs/juzhong-ccfe-none-progress.json
rc=$?
printf "%s\n" "$rc" > /data/common/lgt/GroupSketch/experiments/2026-11-08-ccfe-full-ablation/logs/juzhong-ccfe-none-exit-code.txt
exit "$rc"
