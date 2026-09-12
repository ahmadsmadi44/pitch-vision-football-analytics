# Validation, 11 September 2026

## Executed

- Reprocessed all 750 frames of `input_videos/test.mp4` at its native 25 fps using its existing trusted detection cache, including camera translation, team assignment, calibrated coordinates, filtered movement, bounded ball interpolation, ratings, heatmaps, statistics and both rendered videos.
- Retained 31 tracks, with 31 ratings, heatmaps, stats records and source thumbnails. Four previously omitted white-shirt tracks are now assigned through torso sampling across each track's lifespan.
- Rejected 58 motion windows above the configurable 45 km/h threshold. Maximum exported accepted speed: 44.954849 km/h. This threshold is a plausibility filter, not physical-accuracy validation.
- Checked every exported track: identical IDs across files, heatmap samples equal measured-frame counts, sprint duration agrees with rating sprint-frame inputs, ball proximity duration agrees with rating involvement frames, durations stay within the 30-second clip, finite bounded speeds.
- Total assigned ball-proximity time: 13.08 seconds. Long detection gaps remain unknown.
- Fresh YOLO inference on the first source frame produced 29 detections across the model's ball, goalkeeper, player and referee classes.
- Fresh YOLO and ByteTrack on 12 frames completed: 21-23 player-class tracks per frame before downstream filtering, ball detected in 5 frames. Goalkeepers are now retained as players; the strongest ball detection is selected. The 30-second published demo still uses the earlier detection cache, so these tracking changes have only been smoke-tested on fresh frames, not re-evaluated across a full match.
- Eight Python tests passed, covering video seeking and batch streaming, cached operation without inference imports, invalid input, constant movement/FPS, speed rejection and stale-state clearing, occlusion gaps, bounded ball interpolation, possession duration and turnover semantics.
- Node API tests cover stats delivery, unrated-track preservation, unknown/invalid IDs, byte-range video playback, tactical JSON/video delivery, and the S3/Lambda tactical adapter.
- Vite production build covers the portfolio, match explorer, and tactical lab routes.
- Browser checks: 22 lineup buttons, real player details and heatmap, previous/next navigation, team comparison values, Escape closes modal and restores trigger focus, no horizontal desktop overflow. Phone-sized layout inspected and adapted to a vertical pitch.

## Tactical layer

- Evaluated `models/best_topview.pt` and `models/best_topdown_noball.pt` against 12 native-resolution SoccerTrack frames. `best_topview.pt` matched 264 of 264 annotated players within 35 pixels and was the only candidate with ball detections, so retraining was not justified for this phase. Filtering detections to the registered field removes the off-pitch people responsible for most raw false positives.
- Processed all 362 frames of `input_videos/clip_01_0-12s.mp4` at stride 3 and all 900 frames of `archive/top_view/videos/D_20220220_1_0030_0060.mp4` at stride 3. Inference used the trained model and video only; matching annotations were loaded afterwards by the evaluation script.
- Full-pipeline player results at a 35-pixel centre threshold: 2,453/2,662 matches for the first clip (92.15% recall, 98.55% precision) and 6,138/6,600 for the second (93.00% recall, 95.98% precision).
- Assigned-team results: 2,411/2,412 correct for the first clip (99.96%) and 6,033/6,054 for the second (99.65%). Outfield assignment uses temporally aggregated non-grass appearance. Nine goalkeeper source tracks across both clips have reviewed overrides because their red/yellow kits form a third visual class.
- The coherent retained ball path covers 68/121 sampled frames (56.20%) in the first clip and 255/300 (85.00%) in the second. Median calibrated error is 0.059 m and 0.080 m respectively; all retained samples are within one metre of the matching annotations. Static pitch-marking detections and implausible jumps are rejected, while short bounded gaps are interpolated.
- Tactical exports contain formation edges, per-team width/depth/block estimates, phase labels, pressure candidates, nearest-player space cells, open passing lanes, lateral ball reactions, turnover candidates, and frame-specific coaching hypotheses. Missing ball or fewer than seven assigned outfield players produces an unknown label rather than a filled-in result.
- Nine tactical Python tests cover own-goal-relative block labels, missing-player quality gates, motion-required pressure, same-team formation edges, blocked pass lanes, unknown possession, sustained turnovers, static-ball distractors, and identity reassociation across changing source IDs.
- The browser tactical lab was built for synchronized video scrubbing, two team perspectives, shape/space/pass-lane overlays, accessible track selection, event filtering, validation metrics, and reduced-motion behavior.
- Browser interaction verified team switching, overlay toggles, event-to-replay seeking, synchronized playback, both source tabs, and zero console warnings/errors. A 390 × 844 responsive pass reported no horizontal overflow.

## Important boundaries

The source clip is Bundesliga development footage, not PSG versus Bayern. Club names and 4-3-3 slots are illustrative. Player IDs remain anonymous. No actual score, roster, player position, 90-minute statistic, goal, assist, shot or xG is fabricated.

The broadcast model is not validated against physical ground truth. Broadcast zoom/rotation, region-limited calibration, detection noise, repeated IDs and short visibility windows limit interpretation. Ratings are experimental clip activity scores.

The tactical clips have annotation-based detection and team/ball audits, but the 105 × 68 m calibration is still an assumed planar mapping rather than an independently surveyed camera calibration. Canonical IDs can fragment after long occlusions. Possession, transitions, block labels, pressure, and passing options are rules with explicit thresholds. Space is nearest-player ownership, not a probabilistic pitch-control model. Coaching text is a frame-specific hypothesis, not a learned tactic or proof that an action will succeed. Source media redistribution is not verified. AWS is not deployed.

## Reproduction

See `README.md` for processing/import commands and test commands. Broadcast run: `../validation/team-fix-2026-09-11`. Tactical runs: `../validation/tactical` and `../validation/tactical-openplay`. Served copies: `data/matches/bundesliga-demo`, `data/tactics/topview-demo`, and `data/tactics/topview-openplay`.
