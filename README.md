# Pitch Vision, football analytics and tactical intelligence

Pitch Vision turns match footage into an explorable 105 × 68 m tactical view. The current demo processes a five-minute tactical-camera excerpt from Liverpool vs Real Madrid with a dual-model detector, temporal team classification, moving-camera pitch registration, formation-seeded identities, short-gap interpolation, activity ratings, heatmaps, phase labels, block estimates, pressure candidates, passing-lane geometry, and rules-based coaching hypotheses.

## Source layout

The complete reviewed source is browsable in this repository. The React interface lives in `pitch-vision-platform/frontend`, the Express API lives in `pitch-vision-platform/backend`, and the Python tactical pipeline lives in `tactical`. The processed five-minute tactical export, tests, notebooks, and all 22 player portraits are included directly in their relevant folders.

## Run the interface

```powershell
cd pitch-vision-platform
npm install
npm run api

cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173/pitch-vision/tactics/liverpool-madrid-five`.

The repository intentionally omits model weights and match footage. Put a locally licensed clip at `pitch-vision-platform/data/tactics/liverpool-madrid-five/video.mp4` if synchronized replay is needed. The processed `tactical.json` remains included so the interface and analytics can be reviewed without rerunning inference.

## Rebuild the five-minute export

The expensive detector output is checkpointed separately and is not committed. With local weights, footage, and cached detections in place:

```powershell
python tactical/track_dual_broadcast.py --run validation/liverpool-madrid-5min/dual-detection
python tactical/register_broadcast_pitch.py --clip input_videos/liverpool_madrid_5min.mp4 --run validation/liverpool-madrid-5min/dual-detection
python tactical/build_known_match.py --run validation/liverpool-madrid-5min/dual-detection --lineup input_videos/liverpool_madrid_5min-lineup.json --out validation/liverpool-madrid-5min/final
```

## Interpretation boundaries

- Names are seeded from the supplied mirrored 4-3-3 and motion continuity. They remain provisional until jersey-number anchors are manually reviewed.
- Player positions may be interpolated only across detection-bounded gaps of at most 1.2 seconds. The UI can show dim last-known markers for up to 3 seconds; those markers are excluded from analysis.
- The rating is a relative activity score for the visible sample. It is not a full-match grade and does not infer passes, goals, shots, or xG.
- Tactical suggestions are explainable rules triggered by measured geometry. They are coaching hypotheses, not learned strategic recommendations.
- The Liverpool–Madrid footage is excluded because public redistribution rights were not established.

## Validation

```powershell
python -m unittest discover -s tactical/tests -v
cd pitch-vision-platform
npm test
cd frontend
npm run build
```

The liquid-glass WebGL adapter vendors the MIT-licensed `dashersw/liquid-glass-js` implementation; its license is retained beside the source in `frontend/src/vendor/liquid-glass/`.
