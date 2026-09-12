"""Checkpointed dual-model detection for a wide broadcast/tactical camera."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from teams import descriptor


def pitch_contour(frame: np.ndarray) -> tuple[np.ndarray | None, float]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([28, 28, 28]), np.array([96, 255, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0
    contour = max(contours, key=cv2.contourArea)
    coverage = cv2.contourArea(contour) / (frame.shape[0] * frame.shape[1])
    return cv2.convexHull(contour), float(coverage)


def on_pitch(box: np.ndarray, contour: np.ndarray | None, use_center: bool = False) -> bool:
    if contour is None:
        return False
    x = float((box[0] + box[2]) / 2)
    y = float((box[1] + box[3]) / 2 if use_center else box[3] - 1)
    return cv2.pointPolygonTest(contour, (x, y), False) >= 0


def iou(a: np.ndarray, b: np.ndarray) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = max(1.0, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection)
    return float(intersection / union)


def duplicate(box: np.ndarray, accepted: list[dict]) -> bool:
    center = (box[:2] + box[2:]) / 2
    height = box[3] - box[1]
    for item in accepted:
        other = np.asarray(item["bbox"], dtype=float)
        other_center = (other[:2] + other[2:]) / 2
        if iou(box, other) >= 0.18:
            return True
        if np.linalg.norm(center - other_center) <= max(10.0, 0.55 * max(height, other[3] - other[1])):
            return True
    return False


def model_signature(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "modifiedNs": stat.st_mtime_ns}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--stride", type=int, default=12, help="60 fps / 12 = 5 tactical observations per second")
    parser.add_argument("--imgsz", type=int, default=1280)
    args = parser.parse_args()

    from ultralytics import YOLO

    args.out.mkdir(parents=True, exist_ok=True)
    records_path = args.out / "detections.jsonl"
    meta_path = args.out / "detection-meta.json"
    broadcast_path = Path("models/best_broadcast.pt")
    topview_path = Path("models/best_topview.pt")
    cap = cv2.VideoCapture(str(args.clip))
    if not cap.isOpened():
        raise ValueError(f"Cannot open {args.clip}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    metadata = {
        "schemaVersion": 1,
        "clip": str(args.clip),
        "clipSize": args.clip.stat().st_size,
        "fps": fps,
        "frameCount": frame_count,
        "width": width,
        "height": height,
        "stride": args.stride,
        "imgsz": args.imgsz,
        "models": {"broadcast": model_signature(broadcast_path), "topview": model_signature(topview_path)},
    }
    if meta_path.exists():
        existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if existing_meta != metadata:
            raise ValueError("Existing detection checkpoint does not match this clip, stride, image size, or model files")
    else:
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    completed = set()
    if records_path.exists():
        with records_path.open(encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    completed.add(json.loads(line)["frame"])
    print(f"Resuming with {len(completed)} sampled frames already saved", flush=True)

    broadcast = YOLO(broadcast_path)
    topview = YOLO(topview_path)
    summary = {"classes": {}, "sources": Counter(), "sampledFrames": 0, "broadcastBallFrames": 0}
    with records_path.open("a", encoding="utf-8", buffering=1) as sink:
        for frame_index in range(frame_count):
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % args.stride or frame_index in completed:
                continue
            contour, field_coverage = pitch_contour(frame)
            primary = broadcast.predict(frame, imgsz=args.imgsz, conf=0.10, verbose=False)[0]
            secondary = topview.predict(frame, imgsz=args.imgsz, conf=0.10, verbose=False)[0]
            accepted: list[dict] = []
            class_counts = Counter()
            for result, source_name, minimum in ((primary, "broadcast", 0.18), (secondary, "topview-rescue", 0.28)):
                names = result.names
                for box, class_id, confidence in zip(
                    result.boxes.xyxy.cpu().numpy(),
                    result.boxes.cls.cpu().numpy().astype(int),
                    result.boxes.conf.cpu().numpy(),
                ):
                    label = str(names[int(class_id)]).lower()
                    if label not in {"player", "goalkeeper"} or confidence < minimum or not on_pitch(box, contour):
                        continue
                    if source_name == "topview-rescue" and duplicate(box, accepted):
                        continue
                    appearance = descriptor(frame, box.tolist())
                    accepted.append(
                        {
                            "bbox": [round(float(value), 3) for value in box],
                            "confidence": round(float(confidence), 5),
                            "source": source_name,
                            "detectorClass": label,
                            "appearance": appearance.tolist() if appearance is not None else None,
                        }
                    )
                    class_counts[label] += 1

            balls = []
            for box, class_id, confidence in zip(
                primary.boxes.xyxy.cpu().numpy(),
                primary.boxes.cls.cpu().numpy().astype(int),
                primary.boxes.conf.cpu().numpy(),
            ):
                label = str(primary.names[int(class_id)]).lower()
                if label == "ball" and confidence >= 0.10 and on_pitch(box, contour, use_center=True):
                    balls.append({"bbox": [round(float(value), 3) for value in box], "confidence": round(float(confidence), 5)})

            record = {
                "frame": frame_index,
                "time": round(frame_index / fps, 6),
                "fieldCoverage": round(field_coverage, 5),
                "players": accepted,
                "ballCandidates": balls,
            }
            sink.write(json.dumps(record, separators=(",", ":")) + "\n")
            sink.flush()
            os.fsync(sink.fileno())
            completed.add(frame_index)
            summary["sampledFrames"] += 1
            summary["sources"].update(item["source"] for item in accepted)
            summary["classes"][str(frame_index)] = dict(class_counts)
            summary["broadcastBallFrames"] += bool(balls)
            if len(completed) % 25 == 0:
                print(f"Saved {len(completed)} / {(frame_count + args.stride - 1) // args.stride} sampled frames", flush=True)
    cap.release()

    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    source_counts = Counter(item["source"] for record in records for item in record["players"])
    result = {
        **metadata,
        "sampledFrames": len(records),
        "sourceDetections": dict(source_counts),
        "meanPlayersPerFrame": sum(len(record["players"]) for record in records) / max(1, len(records)),
        "ballCandidateFrames": sum(bool(record["ballCandidates"]) for record in records),
        "complete": len(records) == (frame_count + args.stride - 1) // args.stride,
    }
    (args.out / "detection-summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
