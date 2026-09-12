# Pitch Vision, football analytics and tactical intelligence

Pitch Vision turns match footage into an explorable 105 × 68 m tactical view. The current build processes a five-minute Liverpool vs Real Madrid excerpt with dual-model detection, moving-camera pitch registration, formation-seeded identities, short-gap interpolation, activity ratings, heatmaps, phase labels, block estimates, pressure candidates, passing-lane geometry, and rules-based coaching hypotheses.

## Source handoff

Download `pitch-vision-source.zip` from this repository and extract it. The archive contains the complete Git commit, React and Express platform, processed five-minute tactical JSON, Python vision pipeline, tests, notebooks, 22 player portraits, and `CLAUDE_HANDOFF.md` with the portfolio integration instructions.

The match video, model weights, detection cache, and generated training dataset are intentionally excluded.

## Run

From `pitch-vision-platform`, run `npm ci` and `npm run api`. In a second terminal, run `npm ci` and `npm run dev` from `pitch-vision-platform/frontend`.

Open `http://127.0.0.1:5173/pitch-vision/tactics/liverpool-madrid-five`.

## Interpretation boundaries

Player names are formation-seeded and provisional. Ratings measure visible activity in this five-minute excerpt. Dim held markers improve visual continuity but are excluded from analysis. Tactical suggestions are explainable geometry rules, not learned coaching recommendations. The current build does not claim passes, shots, goals, or xG.
