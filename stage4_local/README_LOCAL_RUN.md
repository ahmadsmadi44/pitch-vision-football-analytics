# Pitch Vision — Stage 4 (Analytics Layer), local GPU run

This is the same pipeline as `pitch_vision_stage4_analytics_v2.ipynb`, packaged to run on
your own machine instead of Colab — for when Colab's free GPU quota is used up (it resets
after roughly a day, but this doesn't wait on that).

## Why not run it through Claude directly?

Claude's connection to this computer runs inside an isolated sandbox with no access to
this machine's actual GPU or graphics driver — it can read/write files in the connected
folder, but it can't run CUDA code on your real hardware. This has to run in your own
terminal, on your own machine, where your GPU driver is actually installed.

## One-time setup

1. Open a terminal (Anaconda Prompt, if you have Anaconda — this machine looks like it
   does) in this folder.
2. Check your GPU's CUDA version: `nvidia-smi` (top-right corner shows "CUDA Version").
3. Install a matching CUDA build of PyTorch — go to
   https://pytorch.org/get-started/locally/, pick your OS / package manager / CUDA
   version, and run the exact command it gives you. Skipping this and just running
   `pip install torch` often silently installs a CPU-only build, which won't error —
   it'll just make everything below run 20-40x slower with no warning, the same trap
   the Colab notebook can fall into if its GPU runtime doesn't actually attach.
4. `pip install -r requirements.txt`

## If the camera pans (broadcast footage)

The pipeline computes ONE homography from frame 0's calibration and reuses it for the
whole clip — valid for a static overhead shot, not for a camera that pans/zooms partway
through, because `CameraMovementEstimator` only corrects pixel-level *translation*; a
real broadcast pan is usually the camera *rotating*, which changes the actual pixel-to-
pitch mapping in a way translation-correction can't fix. If your clip pans, don't
calibrate on the full clip — trim it down to whichever stretch holds the camera framing
steadiest, and only trust results from that stretch.

`find_static_segment.py` finds that stretch for you instead of eyeballing the video:
```
python find_static_segment.py --clip path\to\your_clip.mp4 --window 6
```
It scans the clip (streamed frame-by-frame, doesn't load it all into memory) using the
same background-feature optical-flow signal `CameraMovementEstimator` uses, reports the
calmest N-second window, and prints an `ffmpeg` command to trim to exactly that window.
Run the ffmpeg command it gives you, then calibrate and run on the **trimmed** clip, not
the original. Needs `ffmpeg` on your PATH to actually do the trim — `winget install ffmpeg`
or https://ffmpeg.org/download.html if `ffmpeg -version` doesn't work in your terminal.

## Running it

Two steps — calibration needs a human eye, same as every notebook in this project so far.

**Step 1 — calibrate** (fast, no GPU or model needed):
```
python run_local.py --clip path\to\your_clip.mp4 --mode calibrate
```
This saves `frame0_grid.png` next to this script. Open it and read off the pixel (x, y)
of any 4 points whose real-world pitch position you know and that are on the pitch
surface itself (not all in one line) — they don't have to be the outer corners. If the
shot doesn't show all 4 corners, use whatever markings ARE fully visible instead: a
penalty box's 4 corners (16.5m x 40.3m from the goal line), or the halfway line's two
touchline ends plus the two points where the center circle (9.15m radius) crosses it, etc.
Go around them in the same consistent order (e.g. top-left, top-right, bottom-right,
bottom-left of whichever rectangle/points you picked). This step also creates
`calibration_config.py` with placeholder values on first run. Edit `PIXEL_CORNERS` there
with what you read off, and `PITCH_LENGTH_M`/`PITCH_WIDTH_M` to match the real-world
dimensions between the points you actually used (not always the full 105 x 68m pitch).

**Step 2 — full run:**
```
python run_local.py --model path\to\best_broadcast.pt --clip path\to\your_clip.mp4 --mode full
```
Produces, next to this script:
- `stage4_broadcast_output.mp4` — the annotated tracking video. **Check this first** —
  do the ellipses track real players, are team colors right, do speeds look plausible?
- `heatmaps.json` — Stage 4.1, per-player position density grids.
- `heatmaps_preview.png` — a quick visual sanity check of the top 4 most-tracked players.
- `ratings.json` — Stage 4.2, the 1–10 player ratings, also printed to the console.

## What's in this folder

The same pipeline modules from the Colab notebooks, unit-tested against hand-computed
ground truth before being packaged here — `utils/`, `trackers/`, `team_assigner/`,
`player_ball_assigner/`, `camera_movement_estimator/`, `view_transformer/`,
`speed_and_distance_estimator/`, `heatmap/`, `player_rating/` — plus `run_local.py`, the
driver script that replaces Colab's upload/download cells with local file paths and
streams the annotated video to disk instead of building extra full copies of the clip in
memory (the same fix that resolved the "used all available RAM" crash in the Colab
version).

## Known limitations (same as the Colab version)

- The pixel-corner calibration is specific to this exact clip's camera framing — a
  different clip needs its own `--mode calibrate` pass.
- Goalkeepers may get misclassified into the wrong team by shirt-color clustering —
  not fixed automatically; a manual override is the likely path.
- The player-rating model's weights (25/20/30/25) are a stated starting point, not a
  settled formula — see `player_rating/player_rating.py`'s own docstring.
- If this clip isn't 25fps, edit the `fps=25` in `SpeedAndDistanceEstimator(...)` inside
  `run_local.py` before trusting speed/distance numbers.
