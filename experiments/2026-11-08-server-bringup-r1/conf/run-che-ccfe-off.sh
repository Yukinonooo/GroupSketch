#!/usr/bin/env bash
set -uo pipefail
cd /data/common/lgt/GroupSketch
source tools/env.sh
export CUDA_VISIBLE_DEVICES=3
python -u animate_svg.py --target experiments/2026-11-08-server-bringup-r1/inputs/che/input --caption 'A car moves forward across the zebra crossing while the traffic light pole stays still.' --num_iter 1000 --num_frames 16 --seed 1000 --timeE --output_folder 2026-11-08-server-bringup-r1/che-ccfe-off --progress_callback_file /data/common/lgt/GroupSketch/experiments/2026-11-08-server-bringup-r1/logs/che-ccfe-off-progress.json
rc=$?
printf "%s\n" "$rc" > /data/common/lgt/GroupSketch/experiments/2026-11-08-server-bringup-r1/logs/che-ccfe-off-exit-code.txt
exit "$rc"
