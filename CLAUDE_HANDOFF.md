# Portfolio integration handoff

The finished Pitch Vision portfolio experience is the single route:

`/pitch-vision/tactics/liverpool-madrid-five`

Link the portfolio card titled **Football Match Analytics** directly to that route. Do not add a separate Pitch Vision case-study route or expose the earlier 12-second and 30-second validation clips.

## What is ready

- `pitch-vision-platform/frontend/src/TacticalLab.jsx` renders the synchronized tactical board, team perspective controls, formation graph, controlled-space overlay, pass-lane candidates, timeline, persistent coaching hypotheses, ratings, player portraits, movement heatmaps, and event list.
- `pitch-vision-platform/frontend/src/LiquidGlass.jsx` is the React adapter for the liquid-glass surface used on the main controls and tactical insight card.
- `pitch-vision-platform/data/tactics/liverpool-madrid-five/tactical.json` is the processed five-minute export and can be reviewed without running the computer-vision models.
- `pitch-vision-platform/frontend/public/assets/players/` contains the 22 portrait assets used by the UI.
- `pitch-vision-platform/frontend/src/App.jsx` redirects all legacy Pitch Vision URLs to the single finished experience.

## Local review

From `pitch-vision-platform`, run `npm install` and `npm run api`. In a second terminal, run `npm install` and `npm run dev` from `pitch-vision-platform/frontend`. Open `http://127.0.0.1:5173/pitch-vision/tactics/liverpool-madrid-five`.

The original match video and model weights are intentionally absent from source control. The JSON analytics and every interface state still work; only synchronized footage playback needs the local file described in the main README.

## Claims to preserve

Names are formation-seeded and provisional. The score is a relative visible-activity rating. Short bounded interpolation and dim last-known markers improve continuity, while held markers remain excluded from analysis. Coaching text is a transparent rules engine over measured geometry. The current build does not claim passes, shots, goals, xG, or commercial-provider parity.
