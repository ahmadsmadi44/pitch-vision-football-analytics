"""Render source and normalized-pitch contact sheets for a named match export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def text(image, value, point, color=(255, 255, 255), scale=.45):
    cv2.putText(image, value, point, cv2.FONT_HERSHEY_SIMPLEX, scale, (12, 18, 18), 3, cv2.LINE_AA)
    cv2.putText(image, value, point, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def pitch_panel(frame, teams):
    image = np.full((540, 960, 3), (38, 91, 50), dtype=np.uint8)
    def pt(x, y): return int(45 + x / 105 * 870), int(35 + y / 68 * 470)
    cv2.rectangle(image, pt(0, 0), pt(105, 68), (235, 239, 230), 2)
    cv2.line(image, pt(52.5, 0), pt(52.5, 68), (235, 239, 230), 2)
    cv2.circle(image, pt(52.5, 34), int(9.15 / 105 * 870), (235, 239, 230), 2)
    for player in frame["players"]:
        color = teams[player["team"]]
        center = pt(player["x"], player["y"])
        cv2.circle(image, center, 8, color, -1, cv2.LINE_AA)
        text(image, f'{player["number"]} {player["name"].split()[-1]}', (center[0] + 9, center[1] - 7), scale=.38)
    if frame["ball"]:
        cv2.circle(image, pt(*frame["ball"]), 5, (30, 220, 255), -1)
    text(image, f'{frame["time"]:.1f}s  possession {frame.get("possessionTeam") or "unknown"}', (50, 25), scale=.55)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.data.read_text(encoding="utf-8"))
    registrations = json.loads(args.registration.read_text(encoding="utf-8"))["frames"]
    reg_by_frame = {item["frame"]: np.asarray(item["imageToPitch"], dtype=float) for item in registrations}
    team_colors = {1: (50, 55, 210), 2: (238, 238, 238)}
    targets = [0, 45, 90, 135, 180, 225, 270, 315]
    frames = [min(payload["frames"], key=lambda item: abs(item["time"] - target)) for target in targets]
    capture = cv2.VideoCapture(str(args.video))
    source_panels, pitch_panels = [], []
    for frame in frames:
        source_frame = frame["frame"]
        capture.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
        ok, image = capture.read()
        if not ok: continue
        inverse = np.linalg.inv(reg_by_frame[source_frame])
        for player in frame["players"]:
            point = cv2.perspectiveTransform(np.float32([[[player["x"], player["y"]]]]), inverse)[0, 0]
            x, y = int(point[0]), int(point[1])
            color = team_colors[player["team"]]
            cv2.circle(image, (x, y), 10, color, 3, cv2.LINE_AA)
            text(image, f'{player["number"]} {player["name"].split()[-1]}', (x + 12, y - 8), scale=.5)
        if frame["ball"]:
            point = cv2.perspectiveTransform(np.float32([[frame["ball"]]]), inverse)[0, 0]
            cv2.circle(image, tuple(np.int32(point)), 8, (30, 220, 255), 3)
        text(image, f'{frame["time"]:.1f}s', (24, 48), scale=1.0)
        source_panels.append(cv2.resize(image, (960, 540)))
        pitch_panels.append(pitch_panel(frame, team_colors))
    capture.release()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, panels in (("identity-source-audit.jpg", source_panels), ("identity-pitch-audit.jpg", pitch_panels)):
        sheet = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:4]), np.hstack(panels[4:6]), np.hstack(panels[6:8])])
        cv2.imwrite(str(args.out / name), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(args.out / name)


if __name__ == "__main__":
    main()
