"""
Two-stage face detection + blur 

Stage 1 (recall-oriented): deface's CenterFace at a low threshold, network
    input capped to --stage1-max-dim (default 1920x1080) regardless of
    source resolution -- avoids the >28s/frame slowdown on 5312x4648 exo
    clips; boxes are still reported in full source-resolution coordinates.
Confidence routing: if stage 1 is already confident (score >=
    --stage1-high-conf), trust it directly and skip stage 2 -- stops stage 2
    from overruling a face stage 1 was already sure about.
Stage 2 (precision-oriented): SCRFD (insightface det_10g) re-checks any
    stage 1 candidate that wasn't already confident, as a second opinion.

Output per clip: debug_frames/ (color-coded boxes: green=final face,
red=stage2-rejected) and blurred_frames/ (the actual anonymized result,
ellipse-blur like deface's own default style).

Usage:
    python scripts/cascade_two_stage_blur.py clips_10s/ego1_sec_SEC001_10s.mp4
    python scripts/cascade_two_stage_blur.py clips_10s/*.mp4
"""
import argparse
import csv
import os

import cv2
import numpy as np
import skimage.draw
from deface.centerface import CenterFace
from insightface.model_zoo import model_zoo
from insightface.utils import storage


def make_stage2_detector(det_thresh):
    """Loads SCRFD (stage 2's face verifier) once. Downloads the buffalo_l
    model pack on first run if it isn't already cached under ~/.insightface."""
    model_dir = storage.ensure_available("models", "buffalo_l", root="~/.insightface")
    det = model_zoo.get_model(os.path.join(model_dir, "det_10g.onnx"))
    det.prepare(ctx_id=0, det_thresh=det_thresh, input_size=(320, 320))
    return det


def crop_with_margin(frame_bgr, x1, y1, x2, y2, margin=0.3):
    """Cuts a stage-1 candidate box out of the frame, padded by `margin` on
    each side so stage 2 sees a bit of surrounding context, not just a tight crop."""
    h, w = frame_bgr.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    mx, my = bw * margin, bh * margin
    cx1 = max(0, int(x1 - mx))
    cy1 = max(0, int(y1 - my))
    cx2 = min(w, int(x2 + mx))
    cy2 = min(h, int(y2 + my))
    return frame_bgr[cy1:cy2, cx1:cx2]


def stage2_confirms(detector, crop_bgr):
    """Runs SCRFD on a single crop and reports whether it independently
    agrees there's a face in it, plus its confidence score."""
    if crop_bgr.size == 0:
        return False, 0.0
    bboxes, _ = detector.detect(crop_bgr)
    if bboxes is None or len(bboxes) == 0:
        return False, 0.0
    return True, float(max(bboxes[:, 4]))


def iter_frames(video_path, stride=1):
    """Reads a video and yields every `stride`-th frame (e.g. stride=5 ->
    every 5th frame), along with its frame index and timestamp in seconds."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            yield idx, idx / fps if fps else idx, frame_bgr
        idx += 1
    cap.release()


def draw(frame_bgr, x1, y1, x2, y2, label, color):
    """Draws a labeled debug box on a frame (used for the color-coded
    green/red review output, not the actual blurred result)."""
    cv2.rectangle(frame_bgr, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
    cv2.putText(frame_bgr, label, (int(x1), max(0, int(y1) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def blur_region_ellipse(frame_bgr, x1, y1, x2, y2, blur_factor=2):
    """Same technique deface uses: heavy box-blur, masked to an ellipse so
    there's no hard rectangular edge in the final video."""
    h, w = frame_bgr.shape[:2]
    x1, x2 = max(0, x1), min(w - 1, x2)
    y1, y2 = max(0, y1), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame_bgr[y1:y2, x1:x2]
    ksize = (max(1, (x2 - x1) // blur_factor), max(1, (y2 - y1) // blur_factor))
    blurred = cv2.blur(roi, ksize)
    ey, ex = skimage.draw.ellipse((y2 - y1) // 2, (x2 - x1) // 2, (y2 - y1) // 2, (x2 - x1) // 2,
                                   shape=roi.shape[:2])
    roi[ey, ex] = blurred[ey, ex]
    frame_bgr[y1:y2, x1:x2] = roi


def process_video(video_path, args, stage1, stage2):
    """Runs the full cascade on one video: for every sampled frame, gets
    stage-1 candidates, routes each through confidence-based auto-confirm or
    stage 2, then writes the debug frame, the blurred frame, and a CSV row
    per candidate. This is the main per-clip driver."""
    base = os.path.splitext(os.path.basename(video_path))[0]
    debug_dir = os.path.join(args.outdir, base, "debug_frames")
    blur_dir = os.path.join(args.outdir, base, "blurred_frames")
    for d in (debug_dir, blur_dir):
        os.makedirs(d, exist_ok=True)

    csv_path = os.path.join(args.outdir, base, "cascade_results.csv")
    n_stage1 = n_auto_high_conf = n_stage2_confirmed = n_final = 0

    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame_idx", "t_sec", "x1", "y1", "x2", "y2",
                          "stage1_score", "auto_confirmed_high_stage1",
                          "stage2_confirmed", "stage2_score", "final_confirmed"])

        for idx, t_sec, frame_bgr in iter_frames(video_path, stride=args.stride):
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            dets, _ = stage1(frame_rgb, threshold=args.stage1_thresh)

            debug_frame = frame_bgr.copy()
            blurred_frame = frame_bgr.copy()

            for (x1, y1, x2, y2, s1_score) in dets:
                n_stage1 += 1
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                auto_confirmed = s1_score >= args.stage1_high_conf
                s2_confirmed, s2_score, final_confirmed = False, 0.0, False

                if auto_confirmed:
                    n_auto_high_conf += 1
                    final_confirmed = True
                    n_final += 1
                else:
                    crop = crop_with_margin(frame_bgr, x1, y1, x2, y2, margin=args.margin)
                    s2_confirmed, s2_score = stage2_confirms(stage2, crop)
                    if s2_confirmed:
                        n_stage2_confirmed += 1
                        final_confirmed = True
                        n_final += 1

                writer.writerow([idx, f"{t_sec:.2f}", x1, y1, x2, y2,
                                  f"{s1_score:.4f}", int(auto_confirmed),
                                  int(s2_confirmed), f"{s2_score:.4f}", int(final_confirmed)])

                if final_confirmed:
                    label = f"FACE (auto {s1_score:.2f})" if auto_confirmed else f"FACE {s1_score:.2f}/{s2_score:.2f}"
                    draw(debug_frame, x1, y1, x2, y2, label, (0, 255, 0))
                    blur_region_ellipse(blurred_frame, x1, y1, x2, y2)
                else:
                    draw(debug_frame, x1, y1, x2, y2, f"STAGE2-REJECTED {s1_score:.2f}", (0, 0, 255))

            cv2.imwrite(os.path.join(debug_dir, f"frame_{idx:06d}.jpg"), debug_frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
            cv2.imwrite(os.path.join(blur_dir, f"frame_{idx:06d}.jpg"), blurred_frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])

    print(f"[{base}] stage1 candidates: {n_stage1}  "
          f"auto-confirmed (high stage1 conf): {n_auto_high_conf}  "
          f"stage2-confirmed: {n_stage2_confirmed}  "
          f"FINAL confirmed faces: {n_final}")
    print(f"[{base}] debug frames -> {debug_dir}")
    print(f"[{base}] blurred frames -> {blur_dir}")


def main():
    """Parses CLI args, loads stage 1 (CenterFace) and stage 2 (SCRFD) once,
    then processes every video path passed in, one at a time."""
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--stage1-thresh", type=float, default=0.1)
    ap.add_argument("--stage1-high-conf", type=float, default=0.5,
                     help="stage 1 score at/above which we trust it directly and skip stage 2 "
                          "entirely (stops stage 2 from overruling a confident stage 1 detection)")
    ap.add_argument("--stage2-thresh", type=float, default=0.3)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--stage1-max-dim", type=int, nargs=2, default=(1920, 1080), metavar=("W", "H"))
    ap.add_argument("--outdir", default="cascade_two_stage_out")
    args = ap.parse_args()

    w, h = args.stage1_max_dim
    stage1 = CenterFace(backend="auto", in_shape=(w, h))
    stage2 = make_stage2_detector(args.stage2_thresh)

    for video_path in args.videos:
        process_video(video_path, args, stage1, stage2)


if __name__ == "__main__":
    main()
