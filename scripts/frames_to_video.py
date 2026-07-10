"""
Stitch a folder of frame_XXXXXX.jpg images back into a single playable
video. Used for two purposes:
    - blurred_frames/  -> the actual anonymized output video
    - debug_frames/    -> the color-coded review video (green = confirmed
      face, red = rejected), so you can watch accept/reject decisions over
      time instead of scrolling through images one by one.

Usage:
    python scripts/frames_to_video.py cascade_two_stage_out/ego1_sec_SEC001_10s/blurred_frames \
        -o final_output_videos/ego1_sec_SEC001_10s.mp4 --fps 6

Notes:
    --fps should roughly match (source video fps) / (--stride used when running
    cascade_two_stage_blur.py) if you want real-time playback speed. E.g. a 30fps
    source clip processed with --stride 5 -> use --fps 6.
"""
import argparse
import glob
import os

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames_dir", help="directory of frame_XXXXXX.jpg images (e.g. .../frames)")
    ap.add_argument("-o", "--output", default=None, help="output mp4 path (default: <frames_dir>/../review.mp4)")
    ap.add_argument("--fps", type=float, default=6.0)
    args = ap.parse_args()

    frame_paths = sorted(glob.glob(os.path.join(args.frames_dir, "frame_*.jpg")))
    if not frame_paths:
        raise SystemExit(f"No frame_*.jpg files found in {args.frames_dir}")

    first = cv2.imread(frame_paths[0])
    h, w = first.shape[:2]

    out_path = args.output or os.path.join(args.frames_dir, "..", "review.mp4")
    out_path = os.path.normpath(out_path)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, args.fps, (w, h))

    for p in frame_paths:
        frame = cv2.imread(p)
        if frame is None:
            continue
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
        writer.write(frame)

    writer.release()
    print(f"wrote {len(frame_paths)} frames -> {out_path}")
    print(f"done -> {out_path}")


if __name__ == "__main__":
    main()
