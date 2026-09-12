#!/usr/bin/env python3
"""
Local (non-Colab) driver for Pitch Vision Stage 4 (Analytics Layer) -- runs the exact
same pipeline as pitch_vision_stage4_analytics_v2.ipynb, but on your own machine/GPU
instead of Colab, reading local files instead of Colab's upload/download dialogs.

SETUP (one-time, in your own terminal / Anaconda Prompt -- NOT through Claude):
    1. Create/activate a Python environment, then install a CUDA-enabled PyTorch that
       matches your GPU driver -- go to https://pytorch.org/get-started/locally/,
       pick your OS/CUDA version, and run the exact command it gives you. Run
       `nvidia-smi` first if you don't know your driver's CUDA version.
       (`pip install torch` with no extra flags often silently installs a CPU-only
       build, which is the same "silently slow" trap the Colab notebook can hit.)
    2. pip install -r requirements.txt

USAGE (two steps -- calibration needs a human eye, same as the notebook):

    Step 1 -- calibrate (fast, no GPU/model needed):
        python run_local.py --clip path\\to\\your_clip.mp4 --mode calibrate
    Saves frame0_grid.png next to this script. Open it, read off the pixel (x, y) of
    4 known pitch points (all 4 corners, or the penalty box corners if that's all
    that's visible), going around the boundary in order -- top-left, top-right,
    bottom-right, bottom-left. This step also creates calibration_config.py with
    placeholder values -- edit PIXEL_CORNERS (and PITCH_LENGTH_M/PITCH_WIDTH_M if you
    used a different reference than the full pitch) before Step 2.

    Step 2 -- full run:
        python run_local.py --model path\\to\\best_broadcast.pt --clip path\\to\\your_clip.mp4 --mode full
    Produces, next to this script: stage4_broadcast_output.mp4 (sanity-check this
    FIRST), heatmaps.json, heatmaps_preview.png, ratings.json.
"""
import argparse
import gc
import importlib
import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

import cv2
import numpy as np
import pandas as pd

from utils.video_utils import VideoFrames, get_native_frame, render_video_streaming
from trackers.tracker import Tracker
from team_assigner.team_assigner import TeamAssigner
from player_ball_assigner.player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator.camera_movement_estimator import CameraMovementEstimator
from view_transformer.view_transformer import ViewTransformer
from speed_and_distance_estimator.speed_and_distance_estimator import SpeedAndDistanceEstimator
from heatmap.heatmap import PositionHeatmapBuilder
from player_rating.player_rating import PlayerRatingModel
from platform_export import export_replay, draw_portfolio_frame
from player_stats import build_player_stats

CALIBRATION_CONFIG_PATH = os.path.join(_SCRIPT_DIR, "calibration_config.py")

DEFAULT_CALIBRATION_CONFIG = '''\
# Edit these after looking at frame0_grid.png (produced by --mode calibrate).
# Go around the boundary in order: top-left, top-right, bottom-right, bottom-left.
# These are PLACEHOLDER values -- the pipeline will silently produce garbage
# heatmaps/ratings/speeds if you run --mode full without replacing them.
PIXEL_CORNERS = [
    [100, 100],
    [1800, 100],
    [1800, 1000],
    [100, 1000],
]

# Real-world dimensions (meters) of whatever 4 points PIXEL_CORNERS actually marks.
# FIFA full-pitch standard is 105 x 68. If you used the penalty box instead, it's
# 40.3 x 16.5. Also used as the heatmap grid's extent, so leave this at the full
# pitch (105 x 68) unless REAL_CORNERS below is also left unset.
PITCH_LENGTH_M = 105
PITCH_WIDTH_M = 68

# Optional: if your 4 PIXEL_CORNERS points are NOT the 4 corners of a single rectangle
# (e.g. you mixed two halfway-line/touchline points with two penalty-box corners,
# because that's what was actually visible), PITCH_LENGTH_M/PITCH_WIDTH_M above aren't
# enough -- they only describe a rectangle. Instead, set REAL_CORNERS to the actual
# (x, y) real-world pitch coordinate of each PIXEL_CORNERS point, in the SAME order,
# using the standard coordinate system x: 0-105 (pitch length), y: 0-68 (pitch width).
# Leave this as None to use the simple rectangle derived from PITCH_LENGTH_M/WIDTH_M
# above instead (the default, for when PIXEL_CORNERS really is one rectangle's corners).
REAL_CORNERS = None
'''


def check_gpu():
    import torch
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)} -- good, inference will use it.")
    else:
        print("*** NO GPU DETECTED by PyTorch -- inference will silently run on CPU instead, ***")
        print("*** which will NOT error, just take ~20-40x longer with no warning.          ***")
        print("Check: does `nvidia-smi` work in this terminal? Does")
        print("`python -c \"import torch; print(torch.__version__)\"` show a version with")
        print("'+cu...' in it (CUDA build) rather than '+cpu'? If it says +cpu, reinstall")
        print("torch using the exact command from https://pytorch.org/get-started/locally/")
        print("for your GPU's CUDA version.")


def run_calibrate(clip_path):
    native_frame0 = get_native_frame(clip_path, 0)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(cv2.cvtColor(native_frame0, cv2.COLOR_BGR2RGB))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
    ax.grid(True, color='yellow', alpha=0.5, linewidth=0.5)
    ax.set_title(f"Native frame 0 ({native_frame0.shape[1]}x{native_frame0.shape[0]}px) -- "
                 f"read off 4 known pitch points' pixel coordinates")
    out_path = os.path.join(_SCRIPT_DIR, "frame0_grid.png")
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path} ({native_frame0.shape[1]}x{native_frame0.shape[0]}px)")

    if not os.path.exists(CALIBRATION_CONFIG_PATH):
        with open(CALIBRATION_CONFIG_PATH, "w") as f:
            f.write(DEFAULT_CALIBRATION_CONFIG)
        print(f"Created {CALIBRATION_CONFIG_PATH} with placeholder values.")
    else:
        print(f"{CALIBRATION_CONFIG_PATH} already exists -- left as-is.")
    print("Open frame0_grid.png, read off 4 known pitch points, then edit PIXEL_CORNERS")
    print("in calibration_config.py before running --mode full.")


def _load_calibration(config_path=None):
    config_path = config_path or CALIBRATION_CONFIG_PATH
    if not os.path.exists(config_path):
        print("No calibration_config.py found -- run --mode calibrate first.")
        sys.exit(1)
    spec = importlib.util.spec_from_file_location("calibration_config", config_path)
    calibration_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(calibration_config)
    real_corners = getattr(calibration_config, "REAL_CORNERS", None)
    return (
        calibration_config.PIXEL_CORNERS,
        calibration_config.PITCH_LENGTH_M,
        calibration_config.PITCH_WIDTH_M,
        real_corners,
    )


def _filter_short_tracks(tracks, min_frames=15):
    """
    Drops any player track_id that's present in fewer than `min_frames` frames total,
    before it ever reaches team assignment, drawing, heatmaps, or ratings.

    Why this exists: after fixing camera-motion compensation (the ORB/RANSAC rewrite
    of CameraMovementEstimator), a full run of test.mp4 still showed ~100 distinct
    track_ids instead of ~22 real players. Diagnosing the actual ratings.json output
    (not another synthetic test) showed a clean split: ~20-30 track_ids with
    substantial, continuous activity (tens of meters covered, dozens of possession
    frames -- a plausible ~22-player set, roughly even team split), and ~65 MORE
    track_ids with LITERALLY zero distance, zero sprints, zero possession, and zero
    recoveries for their entire lifetime. That pattern -- not "some" activity, exactly
    none, ever -- is the signature of a track that existed for only a handful of
    frames: almost certainly a single spurious YOLO detection (bench staff, subs, and
    ball boys near the touchline are visible in this clip and are exactly the kind of
    object briefly misdetected as "player"), not a real player getting fragmented into
    a new ID. This is a detection-quality problem, downstream of tracking association
    entirely -- no amount of further camera-motion or ByteTrack tuning fixes it.

    min_frames=15 (0.6s at 25fps) is a starting point, not a tuned value. run_full()
    prints how many track_ids got kept vs dropped so this is easy to adjust: raise it
    if real players still look over-counted, lower it if the dropped count looks like
    it's cutting into real, brief substitute/cameo appearances. Sanity-check any
    change against stage4_broadcast_output.mp4 itself, not just this printed count.
    """
    frame_counts = {}
    for frame_dict in tracks["players"]:
        for track_id in frame_dict:
            frame_counts[track_id] = frame_counts.get(track_id, 0) + 1

    keep_ids = {tid for tid, count in frame_counts.items() if count >= min_frames}
    dropped = len(frame_counts) - len(keep_ids)

    for frame_dict in tracks["players"]:
        for tid in list(frame_dict.keys()):
            if tid not in keep_ids:
                del frame_dict[tid]

    print(f"Track-length filter: kept {len(keep_ids)} player track_ids (>= {min_frames} frames), "
          f"dropped {dropped} short-lived track_ids (likely spurious detections, not real players).")
    return tracks


def _filter_off_pitch_tracks(tracks, min_in_bounds_frames=15):
    """
    Second filter pass, run AFTER view_transformer.transform_tracks() (so
    position_transformed exists), on top of _filter_short_tracks() above.

    Why this exists as a SEPARATE filter, not just a higher _filter_short_tracks
    threshold: diagnosing test.mp4's actual output after the length filter (44
    track_ids kept, min_frames=15) showed a handful of track_ids tracked for MANY
    frames (so they pass the length filter easily) but with essentially zero
    distance/sprints/possession -- e.g. one track present in nearly every frame with
    exactly 0.0m distance. That profile doesn't match "brief spurious detection" (the
    length filter's target); it matches something that's continuously and reliably
    tracked but is NOT ON THE PITCH -- almost certainly bench staff, a sub, or a
    coach standing in the technical area for most/all of the clip (visible in this
    clip's extracted frames). Their pixel position is stable and easy to track, so
    they survive the length filter, but they fall outside the calibrated pitch
    polygon (position_transformed is None) essentially every frame -- which is
    exactly why they contribute ~0 to every real-world stat (distance/speed/sprints
    all come from position_transformed). Filtering on "was this track_id ever really
    on the calibrated pitch" catches this in a way that raw track-length structurally
    cannot: a persistent off-pitch object and a persistent on-pitch player look
    identical to a pure frame-count filter.

    Consistent with run_full()'s existing "Frame 0: X/Y players landed inside the
    calibrated pitch region" print -- this applies that same in-bounds check across
    the WHOLE clip, per track_id, not just frame 0.
    """
    inbounds_counts = {}
    for frame_dict in tracks["players"]:
        for track_id, track in frame_dict.items():
            if track.get("position_transformed") is not None:
                inbounds_counts[track_id] = inbounds_counts.get(track_id, 0) + 1

    all_ids = set()
    for frame_dict in tracks["players"]:
        all_ids.update(frame_dict.keys())

    keep_ids = {tid for tid in all_ids if inbounds_counts.get(tid, 0) >= min_in_bounds_frames}
    dropped = len(all_ids) - len(keep_ids)

    for frame_dict in tracks["players"]:
        for tid in list(frame_dict.keys()):
            if tid not in keep_ids:
                del frame_dict[tid]

    print(f"Off-pitch filter: kept {len(keep_ids)} player track_ids (>= {min_in_bounds_frames} frames "
          f"actually inside the calibrated pitch region), dropped {dropped} track_ids that were tracked "
          f"but rarely/never on the pitch (likely bench/staff/subs, not real match players).")
    return tracks


def run_full(model_path, clip_path, out_dir=None, reuse_tracks=False, min_track_frames=15, min_inbounds_frames=15,
             calibration_path=None, team_overrides_path=None):
    out_dir = out_dir or _SCRIPT_DIR
    PIXEL_CORNERS, PITCH_LENGTH_M, PITCH_WIDTH_M, REAL_CORNERS_OVERRIDE = _load_calibration(calibration_path)

    os.makedirs(out_dir, exist_ok=True)
    tracks_stub_path = os.path.join(out_dir, "tracks_cache.pkl")
    using_cache = reuse_tracks and os.path.isfile(tracks_stub_path)
    if reuse_tracks and not using_cache:
        raise FileNotFoundError(f"Requested cached tracks do not exist: {tracks_stub_path}")
    if not using_cache:
        check_gpu()

    video_frames = VideoFrames(clip_path)
    fps = video_frames.fps
    print(f"Opened {len(video_frames)} frames at {fps:g} fps (decoded on demand)")

    # Camera movement is estimated FIRST (needs only the raw frames, not tracks) so it
    # can be handed to the tracker below and used to compensate detections for the
    # camera's own motion BEFORE track association -- on a panning clip, skipping this
    # lets a single fast camera pan fragment one real player into many track_ids (see
    # get_object_tracks' docstring). Harmless on a static camera: camera_movement comes
    # back ~[0,0] every frame, so this is a no-op there.
    camera_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement = camera_estimator.get_camera_movement(video_frames)
    max_movement = max(abs(dx) + abs(dy) for dx, dy in camera_movement)
    print(f"Max per-frame camera movement detected: {max_movement:.1f}px")

    tracker = Tracker(None if using_cache else model_path, fps=fps)
    # stub caching: detection (YOLO inference over all frames) is the slowest step by
    # far (20+ minutes on a GTX 1070). Every prior downstream-only fix (team
    # assignment, filtering, ratings weights, calibration corners...) has forced a
    # full re-run of detection just to test a change that has nothing to do with
    # detection itself. --reuse-tracks skips straight to the cached result from the
    # last full run instead. Only valid for iterating on downstream logic -- if you
    # change the model, the clip, or anything about camera-motion/tracking itself,
    # don't use --reuse-tracks (delete tracks_cache.pkl or just omit the flag to force
    # a fresh detection+tracking pass).
    tracks_stub_path = os.path.join(out_dir, "tracks_cache.pkl")
    tracks = tracker.get_object_tracks(
        video_frames, camera_movement=camera_movement,
        read_from_stub=reuse_tracks, stub_path=tracks_stub_path,
    )
    if any(len(tracks[key]) != len(video_frames) for key in ("players", "ball")):
        raise ValueError("Cached tracks and video have different frame counts; rerun inference")
    print("Detection + tracking done (camera-motion-compensated association, 3s occlusion tolerance)"
          + (" [from cache]" if reuse_tracks else ""))

    tracks = _filter_short_tracks(tracks, min_frames=min_track_frames)

    from ball_gaps import fill_ball_gaps
    tracks["ball"] = fill_ball_gaps(tracks["ball"], max_gap_frames=round(fps * .4))
    print("Ball gaps filled only when bounded and at most 0.4 seconds")

    # Camera-motion compensation and the pitch-position homography now run HERE,
    # before team assignment -- moved up from after team/ball assignment in an
    # earlier version of this file. This isn't just a filter-ordering convenience:
    # diagnosing test.mp4's real output showed a genuine team_assigner.py bug this
    # reordering fixes. Team assignment's KMeans color clustering was previously
    # seeing ALL 44 length-filtered track_ids, including ~13 that were later dropped
    # by _filter_off_pitch_tracks as bench/staff/subs (stably tracked, but rarely on
    # the actual pitch). Those non-player color samples (staff polos, hi-vis vests --
    # not football kit colors at all) were feeding into the SAME 2-cluster KMeans
    # meant to separate the two teams' kit colors, dragging the cluster centers
    # around and pushing some genuinely on-team players far enough from the (now
    # skewed) centers to trip the referee-outlier threshold. Confirmed directly:
    # track_id 101, visually a normal Team 1 player in a normal frame (10.7 km/h,
    # moving with play, same kit as his teammates), was being labeled "referee" and
    # silently dropped from ratings.json (team_assigner writes "referee" for anyone
    # whose color doesn't cleanly match either cluster -- see its own docstring).
    # Running the off-pitch filter BEFORE team assignment means KMeans never sees
    # the non-player color samples in the first place.
    camera_estimator.adjust_positions_to_tracks(tracks, camera_movement)

    native_frame0 = get_native_frame(clip_path, 0)
    scale = native_frame0.shape[1] / video_frames[0].shape[1]

    if REAL_CORNERS_OVERRIDE is not None:
        REAL_CORNERS = REAL_CORNERS_OVERRIDE
        print("Using explicit REAL_CORNERS from calibration_config.py (mixed/non-rectangle reference points).")
    else:
        REAL_CORNERS = [[0, 0], [PITCH_LENGTH_M, 0], [PITCH_LENGTH_M, PITCH_WIDTH_M], [0, PITCH_WIDTH_M]]
    if len(REAL_CORNERS) != len(PIXEL_CORNERS):
        print(f"ERROR: PIXEL_CORNERS has {len(PIXEL_CORNERS)} points but REAL_CORNERS has "
              f"{len(REAL_CORNERS)} -- they must be the same length and in matching order.")
        sys.exit(1)
    view_transformer = ViewTransformer(PIXEL_CORNERS, REAL_CORNERS)
    view_transformer.transform_tracks(tracks, pixel_scale=scale)
    in_bounds = sum(1 for p in tracks["players"][0].values() if p.get("position_transformed") is not None)
    print(f"Frame 0: {in_bounds}/{len(tracks['players'][0])} players landed inside the calibrated pitch region")
    if in_bounds == 0:
        print("WARNING: 0 players in bounds -- calibration_config.py's PIXEL_CORNERS is still")
        print("the placeholder (or misread). Re-run --mode calibrate, check frame0_grid.png, and fix it.")

    tracks = _filter_off_pitch_tracks(tracks, min_in_bounds_frames=min_inbounds_frames)

    def _scale_bbox(bbox, s):
        return [c * s for c in bbox]

    team_assigner = TeamAssigner(view="broadcast")
    n_frames = len(tracks["players"])
    # Sample each track over its own lifespan. Global clip timestamps miss tracks
    # appearing between those timestamps and otherwise classify them from one crop.
    appearances = {}
    for fn, frame_tracks in enumerate(tracks["players"]):
        for tid in frame_tracks:
            appearances.setdefault(tid, []).append(fn)
    sample_requests = {}
    for tid, frame_nums in appearances.items():
        for index in np.linspace(0, len(frame_nums) - 1, 7, dtype=int):
            sample_requests.setdefault(frame_nums[index], set()).add(tid)

    per_player_samples = {}
    for fn in sorted(sample_requests):
        native = get_native_frame(clip_path, fn)
        for player_id in sorted(sample_requests[fn]):
            track = tracks["players"][fn][player_id]
            color = team_assigner.get_player_color(native, _scale_bbox(track["bbox"], scale))
            per_player_samples.setdefault(player_id, []).append(color)

    aggregate_colors = {tid: np.median(np.array(colors), axis=0) for tid, colors in per_player_samples.items()}
    team_assigner.assign_team_colors_from_samples(aggregate_colors)
    if team_overrides_path:
        with open(team_overrides_path, encoding="utf-8") as stream:
            overrides = json.load(stream)
        for tid, team in overrides.items():
            if team not in (1, 2, "referee") or int(tid) not in appearances:
                raise ValueError(f"Invalid team override: {tid}: {team}")
            team_assigner.player_team_dict[int(tid)] = team

    for frame_num, player_track in enumerate(tracks["players"]):
        native_frame = None
        for player_id, track in player_track.items():
            if player_id in team_assigner.player_team_dict:
                team = team_assigner.player_team_dict[player_id]
            else:
                if native_frame is None:
                    native_frame = get_native_frame(clip_path, frame_num)
                team = team_assigner.get_player_team(native_frame, _scale_bbox(track["bbox"], scale), player_id)
            tracks["players"][frame_num][player_id]["team"] = team
            if team == "referee":
                tracks["players"][frame_num][player_id]["team_color"] = (255, 255, 255)
            else:
                tracks["players"][frame_num][player_id]["team_color"] = tuple(
                    int(c) for c in team_assigner.team_colors[team]
                )
    print("Team colors (BGR):", team_assigner.team_colors)

    ball_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks["players"]):
        ball_bbox = tracks["ball"][frame_num].get(1, {}).get("bbox")
        assigned_player = ball_assigner.assign_ball_to_player(player_track, ball_bbox) if ball_bbox else -1
        if assigned_player != -1:
            tracks["players"][frame_num][assigned_player]["has_ball"] = True
            assigned_team = tracks["players"][frame_num][assigned_player].get("team")
            if assigned_team in (1, 2):
                team_ball_control.append(assigned_team)
            else:
                team_ball_control.append(0)
        else:
            team_ball_control.append(0)
    team_ball_control = np.array(team_ball_control)
    print("Possession computed")

    speed_estimator = SpeedAndDistanceEstimator(frame_window=5, fps=fps)
    speed_estimator.add_speed_and_distance(tracks)
    print(f"Speed and distance computed; rejected {speed_estimator.rejected_windows} implausible movement windows")
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as stream:
        json.dump(build_player_stats(tracks, fps), stream, allow_nan=False)
    export_replay(tracks, team_ball_control, fps, out_dir,
                  (PITCH_LENGTH_M, PITCH_WIDTH_M), team_assigner.team_colors)

    output_video_path = os.path.join(out_dir, "stage4_broadcast_output.mp4")
    render_video_streaming(
        video_frames, output_video_path,
        draw_fns=[
            lambda f, n: tracker.draw_frame_annotations(f, n, tracks, team_ball_control),
            lambda f, n: speed_estimator.draw_frame_speed_and_distance(f, n, tracks),
        ],
        fps=fps,
    )
    print(f"Saved {output_video_path} -- sanity-check this before trusting heatmaps/ratings below.")
    render_video_streaming(video_frames, os.path.join(out_dir, "portfolio_preview.mp4"),
                           draw_fns=[lambda f, n: draw_portfolio_frame(f, n, tracks)], fps=fps)
    video_frames.close()
    del video_frames
    gc.collect()

    heatmap_builder = PositionHeatmapBuilder(pitch_length_m=PITCH_LENGTH_M, pitch_width_m=PITCH_WIDTH_M, cell_size_m=2.0)
    player_grids = heatmap_builder.build_player_grids(tracks)
    heatmaps_json = heatmap_builder.to_density_json(player_grids)
    with open(os.path.join(out_dir, "heatmaps.json"), "w") as f:
        json.dump(heatmaps_json, f)
    print(f"Saved heatmaps.json ({len(heatmaps_json)} players)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        top_players = sorted(heatmaps_json.items(), key=lambda kv: kv[1]["n_samples"], reverse=True)[:4]
        if top_players:
            fig, axes = plt.subplots(1, len(top_players), figsize=(5 * len(top_players), 4.5))
            if len(top_players) == 1:
                axes = [axes]
            for ax, (tid, entry) in zip(axes, top_players):
                grid = np.array(entry["density_grid"])
                ax.imshow(grid, origin='lower', extent=[0, PITCH_LENGTH_M, 0, PITCH_WIDTH_M], cmap='hot', aspect='auto')
                ax.set_title(f"Player {tid} (team {entry['team']}, {entry['n_samples']} samples)")
                ax.set_xlabel('pitch length (m)')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "heatmaps_preview.png"), dpi=100)
            plt.close(fig)
            print("Saved heatmaps_preview.png")
    except Exception as e:
        print("Heatmap preview image skipped:", e)

    rating_model = PlayerRatingModel()
    ratings = rating_model.rate_players(tracks, team_ball_control.tolist())
    ranked = sorted(ratings.items(), key=lambda kv: kv[1]["rating_1_10"], reverse=True)
    print(f"{'Player':>8}  {'Team':>4}  {'Rating':>6}  {'Dist(m)':>8}  {'Sprint frames':>13}  {'Poss.frames':>11}  {'Recoveries':>10}")
    for tid, r in ranked:
        s = r["raw_stats"]
        print(f"{tid:>8}  {r['team']:>4}  {r['rating_1_10']:>6.2f}  {s['distance_covered']:>8.1f}  "
              f"{s['sprint_count']:>7}  {s['possession_involvement']:>11}  {s['ball_recoveries']:>10}")
    # Track IDs coming out of the real ByteTrack tracker are numpy.int64, not plain
    # Python int -- json.dump's `default` hook only rescues unserializable VALUES, not
    # dict KEYS, so numpy-typed keys need to be stringified explicitly here (same as
    # heatmap.py's to_density_json already does for its own player-id keys).
    ratings_json_safe = {str(tid): r for tid, r in ratings.items()}
    with open(os.path.join(out_dir, "ratings.json"), "w") as f:
        json.dump(ratings_json_safe, f, default=lambda o: int(o) if hasattr(o, 'item') else o)
    print("Saved ratings.json")
    print("\nRemember: normalization is relative to THIS match's players only, and this rating")
    print("model's weights (25/20/30/25) are a stated starting point -- see player_rating.py's")
    print("own docstring for the full rationale before treating any single number as final.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clip", required=True, help="Path to the broadcast/side-view clip")
    parser.add_argument("--model", help="Path to best_broadcast.pt (required for --mode full)")
    parser.add_argument("--mode", choices=["calibrate", "full"], default="full")
    parser.add_argument("--out-dir", help="Separate output folder for this clip/run")
    parser.add_argument("--calibration", help="Per-clip calibration Python file")
    parser.add_argument("--team-overrides", help="JSON mapping track IDs to 1, 2, or referee")
    parser.add_argument("--reuse-tracks", action="store_true",
                         help="Skip YOLO detection/tracking and reuse tracks_cache.pkl from the last "
                              "--mode full run (fast iteration on team assignment/filtering/ratings/"
                              "calibration only -- NOT valid after changing the model, clip, or "
                              "anything about camera-motion/tracking itself).")
    parser.add_argument("--min-track-frames", type=int, default=15,
                         help="Drop any player track_id present in fewer than this many frames "
                              "(default 15 = 0.6s at 25fps) -- filters out short-lived spurious "
                              "detections. See _filter_short_tracks()'s docstring in this file.")
    parser.add_argument("--min-inbounds-frames", type=int, default=15,
                         help="Drop any player track_id that's ACTUALLY INSIDE the calibrated pitch "
                              "region in fewer than this many frames (default 15) -- filters out "
                              "persistently-tracked but off-pitch objects (bench/staff/subs). See "
                              "_filter_off_pitch_tracks()'s docstring in this file.")
    args = parser.parse_args()

    if args.mode == "calibrate":
        run_calibrate(args.clip)
    else:
        if not args.model:
            print("--model is required for --mode full")
            sys.exit(1)
        run_full(args.model, args.clip, out_dir=args.out_dir, reuse_tracks=args.reuse_tracks,
                 min_track_frames=args.min_track_frames, min_inbounds_frames=args.min_inbounds_frames,
                 calibration_path=args.calibration, team_overrides_path=args.team_overrides)


if __name__ == "__main__":
    main()
