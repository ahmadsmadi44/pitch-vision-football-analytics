# Tactical intelligence pipeline

This package converts overhead football footage into an auditable tactical replay. It uses detections from `best_topview.pt`, camera registration, a pitch homography, temporal team appearance, identity reassociation, and transparent tactical rules. Ground-truth annotations are used only by the two evaluation scripts.

## Modules

- `run_topview.py`: detector, ByteTrack, registration, calibration, team assignment, ball path, reassociation, and export.
- `teams.py`: non-grass Lab appearance samples accumulated over a track; goalkeeper kits can remain unassigned instead of being forced into two clusters.
- `ball_path.py`: coherent trajectory selection with static-distractor and speed rejection plus bounded interpolation.
- `association.py`: reconnects fragmented source IDs using motion, position, and team evidence.
- `analysis.py`: phases, turnovers, shape edges, blocks, pressure, space ownership, pass lanes, lateral reactions, and evidence-bearing coaching hypotheses.
- `evaluate_detectors.py`: model-selection benchmark on sparse annotated frames.
- `evaluate_pipeline.py`: post-inference player/team/ball audit against matching SoccerTrack annotations.
- `publish.py`: combines the model output with the labeled audit and creates the browser data/video assets.

## Interpretation rules

Unknown observations remain unknown. Possession requires sustained player-ball proximity. A transition requires a directly observed change of team ownership. Pressure requires proximity and movement toward the ball. A defensive shape is withheld until at least seven assigned outfield players are visible. Block location is calculated relative to the team's own goal, so the rule works in either attacking direction.

Suggested responses to a block or press are deterministic coaching hypotheses supported by the current frame's geometry. They do not claim a learned policy, causal effect, or guaranteed outcome. The space overlay assigns each grid cell to its nearest player and should not be described as probabilistic pitch control.

## Top-view team limitation

Tiny overhead players expose too little kit area for a single-frame RGB crop. The implemented classifier samples native-resolution upper-body pixels, removes grass-colored pixels in Lab space, aggregates evidence across the track, and checks temporal agreement. White and blue outfield teams separate reliably. Goalkeeper kits are red or yellow and constitute a third class, so reviewed per-clip overrides assign those tracks. The UI identifies reviewed goalkeeper assignments explicitly.
