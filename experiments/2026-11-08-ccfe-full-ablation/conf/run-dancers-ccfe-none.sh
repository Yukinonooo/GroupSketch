#!/usr/bin/env bash
set -uo pipefail
cd /data/common/lgt/GroupSketch
source tools/env.sh
export CUDA_VISIBLE_DEVICES=5
python -u animate_svg.py --target experiments/2026-11-08-server-bringup-r1/inputs/dancers/input --caption 'Two people turn a long jump rope while two people in the middle jump over it.' --num_iter 1000 --num_frames 16 --seed 1000 --output_folder 2026-11-08-ccfe-full-ablation/dancers-ccfe-none --progress_callback_file /data/common/lgt/GroupSketch/experiments/2026-11-08-ccfe-full-ablation/logs/dancers-ccfe-none-progress.json
rc=$?
printf "%s\n" "$rc" > /data/common/lgt/GroupSketch/experiments/2026-11-08-ccfe-full-ablation/logs/dancers-ccfe-none-exit-code.txt
exit "$rc"
