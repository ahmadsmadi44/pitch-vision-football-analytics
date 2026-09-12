# Pitch Vision, football analytics and tactical intelligence

Pitch Vision turns match footage into an explorable 105 × 68 m tactical view. The current build processes a five-minute Liverpool vs Real Madrid excerpt with dual-model detection, moving-camera pitch registration, formation-seeded identities, short-gap interpolation, activity ratings, heatmaps, phase labels, block estimates, pressure candidates, passing-lane geometry, and rules-based coaching hypotheses.

## Complete source

[Download the complete reviewed source archive](./pitch-vision-source.zip). It contains the React and Express platform, processed five-minute tactical JSON, Python computer-vision pipeline, tests, notebooks, all 22 player portraits, and `CLAUDE_HANDOFF.md` with portfolio integration instructions.

- Local source commit: `dc8e279`
- SHA-256: `0525557E51452CE8C8B6F3940F4889EEBCC6CD992C47B163917E8A2D67390C69`
- Extracted size excludes dependencies and generated model artifacts

## Product experience

The React interface presents one focused project route: `/pitch-vision/tactics/liverpool-madrid-five`. It combines synchronized playback, formation graphs, space and pass-lane overlays, a tactical timeline, persistent coaching hypotheses, player ratings, portrait-based navigation, and individual movement heatmaps.

## Architecture

- Python and Ultralytics YOLO for detection, tracking, pitch registration, measurement, and tactical export
- React and Vite for the interactive experience
- Node.js and Express for local or object-storage backed analytics and video APIs
- An MIT-licensed WebGL liquid-glass layer for the primary controls and insight surfaces

## Run locally

Extract the archive. From `pitch-vision-platform`, run `npm ci` and `npm run api`. In a second terminal, run `npm ci` and `npm run dev` from `pitch-vision-platform/frontend`. Open `http://127.0.0.1:5173/pitch-vision/tactics/liverpool-madrid-five`.

## Interpretation boundaries

Player names are formation-seeded and provisional. Ratings measure visible activity in this five-minute excerpt. Dim held markers improve visual continuity but are excluded from analysis. Tactical suggestions are explainable geometry rules, not learned coaching recommendations. The current build does not claim passes, shots, goals, or xG.

The match video, model weights, detection cache, and generated training dataset are excluded from the public release.
