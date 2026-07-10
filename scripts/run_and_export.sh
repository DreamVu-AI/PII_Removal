#!/usr/bin/env bash
# Runs the two-stage face-blur cascade on one or more videos, then stitches
# each clip's blurred frames into a final .mp4, saved under
# final_output_videos/<clip_name>.mp4
#
# Usage:
#   ./scripts/run_and_export.sh clips_10s/*.mp4
#   ./scripts/run_and_export.sh GDM_samples/ego1_sec_SEC001.mp4
#   STRIDE=1 FPS=30 ./scripts/run_and_export.sh GDM_samples/*.mp4   # full-fps final run
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

STRIDE="${STRIDE:-5}"
FPS="${FPS:-6}"
OUTDIR="${OUTDIR:-cascade_two_stage_out}"
FINAL_DIR="final_output_videos"

if [ "$#" -eq 0 ]; then
  echo "Usage: $0 <video1> [video2 ...]"
  exit 1
fi

mkdir -p "$FINAL_DIR"

echo "=== Running two-stage cascade (stride=$STRIDE) on: $* ==="
python scripts/cascade_two_stage_blur.py "$@" --stride "$STRIDE" --outdir "$OUTDIR"

for video in "$@"; do
  clip_name=$(basename "$video")
  clip_name="${clip_name%.*}"
  frames_dir="$OUTDIR/$clip_name/blurred_frames"

  if [ ! -d "$frames_dir" ]; then
    echo "!! No blurred frames found for $clip_name, skipping video export"
    continue
  fi

  echo "=== Building final video for $clip_name (fps=$FPS) ==="
  python scripts/frames_to_video.py "$frames_dir" -o "$FINAL_DIR/${clip_name}.mp4" --fps "$FPS"
done

echo "=== Done. Final videos in $FINAL_DIR/ ==="
ls -la "$FINAL_DIR"
