"""Export a local pipeline run. Pickle input must be this project's trusted cache.

Example:
python scripts/convert_match.py --run ../stage4_local --clip ../input_videos/test.mp4 \
    --id bundesliga-demo --title "Bundesliga, development clip"
"""
import argparse
import json
import pickle
import re
import shutil
import subprocess
from pathlib import Path

import cv2


def convert(run, clip, destination, match_id, title, source_url=None):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", match_id):
        raise ValueError("Invalid match id")
    ratings = json.loads((run / "ratings.json").read_text())
    heatmaps = json.loads((run / "heatmaps.json").read_text())
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise ValueError(f"Cannot open {clip}")
    fps, frame_count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(3)), int(cap.get(4))
    if fps <= 0 or frame_count <= 0:
        cap.release()
        raise ValueError("Invalid video metadata")
    with (run / "tracks_cache.pkl").open("rb") as stream:
        tracks = pickle.load(stream)
    if len(tracks["players"]) != frame_count:
        cap.release()
        raise ValueError("Track cache and clip frame counts differ")
    out = destination / match_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "thumbnails").mkdir(exist_ok=True)
    appearances = {}
    for frame_num, players in enumerate(tracks["players"]):
        for tid, track in players.items():
            if str(tid) in heatmaps or str(tid) in ratings:
                appearances.setdefault(str(tid), []).append((frame_num, track["bbox"]))
    thumbnails = []
    scale = max(1, width / 1920)
    try:
        for tid, samples in appearances.items():
            frame_num, bbox = samples[len(samples) // 2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ok, frame = cap.read()
            if not ok:
                continue
            x1, y1, x2, y2 = [int(v * scale) for v in bbox]
            x1, x2 = max(0, x1), min(width, x2)
            y1, y2 = max(0, y1), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2]
            if cv2.imwrite(str(out / "thumbnails" / f"{tid}.jpg"), crop):
                thumbnails.append(tid)
    finally:
        cap.release()
    for name in ("ratings.json", "heatmaps.json"):
        shutil.copy2(run / name, out / name)
    has_tracks = (run / "tracks.json").exists()
    if has_tracks:
        shutil.copy2(run / "tracks.json", out / "tracks.json")
    summary_path = run / "run_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    has_stats = (run / "stats.json").exists()
    if has_stats:
        shutil.copy2(run / "stats.json", out / "stats.json")
    meta = {
        "schemaVersion": 1, "id": match_id, "title": title,
        "fps": fps, "frameCount": frame_count, "durationSeconds": frame_count / fps,
        "pitch": summary.get("pitch", {"length": 105, "width": 68, "dimensionsVerified": False}),
        "teams": [{"id": i, "name": f"Team {i}", "color": summary.get("teamColors", {}).get(str(i))} for i in (1, 2)],
        "ratedPlayerCount": len(ratings), "trackedPlayerCount": len(heatmaps),
        "thumbnailIds": sorted(thumbnails, key=int), "hasTracks": has_tracks,
        "hasStats": has_stats,
        "source": {"url": source_url, "clip": clip.name, "redistributionVerified": False},
        "ratingWeights": {"distance_covered": .25, "sprint_count": .20, "possession_involvement": .30, "ball_recoveries": .25},
        "limitations": [
            "Short clip: ratings compare visible tracks within this clip, not full-match performance.",
            "Track IDs represent observations, not verified player identities.",
            "Sprint count is a count of frames above 20 km/h, not separate sprint events.",
            "Possession and recoveries are proximity heuristics.",
            "Position estimates use one calibrated region and translation-only camera compensation."
        ]
    }
    video = run / "portfolio_preview.mp4"
    if video.exists():
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg is required to encode a browser-compatible preview")
        subprocess.run([ffmpeg, "-y", "-v", "error", "-i", str(video), "-an", "-c:v", "libx264",
                        "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out / "video.mp4")], check=True)
    meta["hasVideo"] = (out / "video.mp4").exists()
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out.resolve()), "players": len(heatmaps), "rated": len(ratings), "thumbnails": len(thumbnails)}))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--clip", required=True, type=Path)
    parser.add_argument("--destination", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "matches")
    parser.add_argument("--id", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--source-url")
    args = parser.parse_args()
    convert(args.run, args.clip, args.destination, args.id, args.title, args.source_url)
