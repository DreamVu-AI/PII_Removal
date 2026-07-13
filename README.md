# PI Removal — Face Blurring Pipeline

Automatically blurs faces in the factory-floor camera footage (ego/exo/wrist
cameras) while keeping false positives (blurring something that isn't a
face) to a minimum.

## How it works (short version)

Two-stage detection cascade per frame:

1. **Stage 1 — CenterFace** (from the `deface` package): fast, deliberately
   loose face detector. Flags every possible candidate at a low confidence
   threshold, so nothing gets missed.
2. **Stage 2 — SCRFD** (from `insightface`): a stronger, independent face
   verifier. Any Stage-1 candidate that isn't already highly confident gets
   re-checked here before it's trusted.

Only candidates that survive both stages get blurred (ellipse-masked
box-blur, the same style `deface` itself uses). See
`scripts/cascade_two_stage_blur.py` for the full logic and inline comments.

## Key parameter choices (vs. deface's defaults)

This pipeline uses deface's own detector (CenterFace) as Stage 1, not the
plain `deface` CLI with its defaults — a few settings were changed
deliberately:

- **Detection threshold: 0.1** instead of deface's default 0.2. Set low on
  purpose — since every Stage-1 candidate gets independently re-verified by
  SCRFD (Stage 2) before anything is blurred, it's safer to over-flag at
  Stage 1 than risk silently dropping a real face before it ever gets a
  second look.
- **Stage 2 verifier: SCRFD** (from InsightFace), used to confirm or reject
  every Stage-1 candidate before blurring.
- **Detector input capped at 1920×1080**, regardless of the source video's
  native resolution. deface's default runs its network at the frame's full
  native resolution — on this project's exo camera footage (5312×4648) that
  was measured at 28+ seconds *per frame*. Capping the network's input size
  gave a ~35x speedup with no accuracy loss (verified by diffing detection
  output at both settings — identical results, just faster).
- **Blur style**: kept deface's own default (ellipse-masked blur, not a
  hard rectangle or pixelation).

## Setup

```bash
cd PI_Removal
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On first run, `insightface` auto-downloads its SCRFD model (~280MB) to
`~/.insightface/models/buffalo_l/` — one-time, needs internet access once.

## Running on a folder of .mp4 files

```bash
source .venv/bin/activate
./scripts/run_and_export.sh /path/to/input_folder/*.mp4
```

This runs the full pipeline (detect → verify → blur) on every `.mp4` in that
folder and writes one final blurred video per input to `final_output_videos/`,
named after the source file, e.g.:

```
input_folder/ego1_sec_SEC001.mp4  -->  final_output_videos/ego1_sec_SEC001.mp4
```

### Options (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `STRIDE` | `5` | Process every Nth frame. `5` = fast preview (skips 4 of every 5 frames). Use `STRIDE=1` to process every frame for a real final-quality export — much slower, especially on high-resolution exo footage. |
| `FPS` | `6` | Playback framerate of the output video. Should be `source_fps / STRIDE` to keep real-time speed (e.g. a 30fps source at `STRIDE=5` → `FPS=6`; at `STRIDE=1` → `FPS=30`). |
| `OUTDIR` | `cascade_two_stage_out` | Where intermediate per-frame results are saved (see below). |

Example — full-quality final export (every frame, correct playback speed):

```bash
STRIDE=1 FPS=30 ./scripts/run_and_export.sh /path/to/input_folder/*.mp4
```

## Where outputs go

For each input video `<clip>.mp4`, two things get written:

**1. Intermediate results**, under `cascade_two_stage_out/<clip>/` (or
whatever `$OUTDIR` is set to):
- `blurred_frames/` — every processed frame, with detected faces blurred.
  This is what gets stitched into the final video.
- `debug_frames/` — the same frames, but with color-coded boxes instead of
  blur (green = confirmed face, red = rejected by Stage 2) — useful for
  auditing *why* a decision was made, not meant to be the deliverable.
- `cascade_results.csv` — one row per candidate detection, with every
  stage's score and final decision, for anyone who wants the raw numbers.

**2. The final deliverable video**, under `final_output_videos/<clip>.mp4`
— this is the actual anonymized output, built by stitching
`blurred_frames/` back into a video at the framerate you specified.

## Running the two stages directly (without the video-export wrapper)

If you just want the frame-by-frame outputs without building a video:

```bash
python scripts/cascade_two_stage_blur.py path/to/video1.mp4 path/to/video2.mp4 --stride 5
```

See `python scripts/cascade_two_stage_blur.py --help` for all tunable
thresholds (Stage 1 recall threshold, Stage 1 auto-confirm confidence,
Stage 2 threshold, crop margin, etc).

## Notes

- `scripts/test_unused/` holds earlier iterations (MediaPipe-based
  verifier, ArUco marker-veto stage, OCR experiments) kept for reference —
  not part of the active pipeline. A 3-stage version with an extra
  marker-detection stage was tested and found to produce byte-identical
  results to the current 2-stage pipeline on all 7 sample cameras, so it
  was dropped for simplicity.
- Designed and tested on this project's specific camera rig footage
  (ego/exo/wrist cameras with fisheye lenses, calibration markers mounted
  on the rig) — thresholds may need retuning for very different footage.
