"""Compact, measured positions for interactive replay; no synthetic coordinates."""
import json
from pathlib import Path


def export_replay(tracks, possession, fps, out_dir, pitch, team_colors, stride=5):
    frames = []
    for fn in range(0, len(tracks["players"]), stride):
        players = []
        for tid, player in tracks["players"][fn].items():
            pos = player.get("position_transformed")
            if pos is None:
                continue
            players.append({
                "id": str(tid), "team": player.get("team"),
                "x": round(float(pos[0]), 3), "y": round(float(pos[1]), 3),
                "speed": player.get("speed"), "distance": player.get("distance"),
                "hasBall": bool(player.get("has_ball")),
            })
        ball = tracks["ball"][fn].get(1, {}).get("position_transformed")
        frames.append({"frame": fn, "time": fn / fps, "players": players,
                       "ball": ball, "possessionTeam": int(possession[fn])})
    payload = {"schemaVersion": 1, "fps": fps, "stride": stride, "frames": frames}
    Path(out_dir, "tracks.json").write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
    ids = {str(tid) for frame in tracks["players"] for tid in frame}
    summary = {"fps": fps, "frameCount": len(tracks["players"]), "retainedTracks": len(ids),
               "pitch": {"length": pitch[0], "width": pitch[1], "dimensionsVerified": False},
               "teamColors": {str(k): [int(c) for c in v[::-1]] for k, v in team_colors.items()},
               "positionModel": "single calibration with translation-only camera compensation"}
    Path(out_dir, "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def draw_portfolio_frame(frame, frame_num, tracks):
    """A light annotation pass: team-colored ground rings, with no diagnostic text."""
    import cv2
    for player in tracks["players"][frame_num].values():
        if player.get("team") not in (1, 2):
            continue
        x1, _, x2, y2 = player["bbox"]
        color = player.get("team_color", (255, 255, 255))
        radius = max(5, int((x2 - x1) * .65))
        center = (int((x1 + x2) / 2), int(y2))
        cv2.ellipse(frame, center, (radius, max(2, int(radius * .28))), 0, 0, 360,
                    color, 2, cv2.LINE_AA)
    return frame
