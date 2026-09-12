"""Publish a validated tactical run into the React/Node data contract.

The tactical pipeline never reads ground-truth annotations during inference.  The
optional quality audit is attached afterwards so the UI can distinguish model
output from evaluation evidence.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--id", required=True)
    parser.add_argument(
        "--destination",
        default=Path("pitch-vision-platform/data/tactics"),
        type=Path,
    )
    args = parser.parse_args()

    tactical_path = args.run / "tactical.json"
    audit_path = args.run / "quality-audit.json"
    if not tactical_path.is_file():
        raise FileNotFoundError(f"Missing tactical export: {tactical_path}")
    if not audit_path.is_file():
        raise FileNotFoundError(f"Run evaluate_pipeline.py first: {audit_path}")
    if not args.video.is_file():
        raise FileNotFoundError(f"Missing source video: {args.video}")

    payload = json.loads(tactical_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    payload["validation"] = audit
    payload["validationMethod"] = (
        "Compared after inference with matching SoccerTrack annotations at a "
        "35-pixel player-centre threshold. Annotations are never model inputs."
    )

    target = args.destination / args.id
    target.mkdir(parents=True, exist_ok=True)
    (target / "tactical.json").write_text(
        json.dumps(payload, separators=(",", ":")), encoding="utf-8"
    )

    encoded = target / "video.mp4"
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(args.video),
        "-vf",
        "scale=1280:-2",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "24",
        "-movflags",
        "+faststart",
        "-an",
        str(encoded),
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as error:
        raise RuntimeError("ffmpeg is required to publish browser-compatible video") from error

    print(f"Published {args.id} to {target}")


if __name__ == "__main__":
    main()
