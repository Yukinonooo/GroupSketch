#!/usr/bin/env bash
set -uo pipefail
cd /data/common/lgt/GroupSketch
source tools/env.sh
export CUDA_VISIBLE_DEVICES=2
python -u animate_svg.py --target experiments/2026-11-08-server-bringup-r1/inputs/che/input --caption 'A car moves forward across the zebra crossing while the traffic light pole stays still.' --num_iter 1000 --num_frames 16 --seed 1000 --timeE --output_folder 2026-11-08-server-bringup-r1/che-ccfe-on --progress_callback_file /data/common/lgt/GroupSketch/experiments/2026-11-08-server-bringup-r1/logs/che-ccfe-on-progress.json --dino
rc=$?
printf "%s\n" "$rc" > /data/common/lgt/GroupSketch/experiments/2026-11-08-server-bringup-r1/logs/che-ccfe-on-exit-code.txt
exit "$rc"
