"""Visual domain check for the broadcast and top-view detectors on a new clip.

This is a diagnostic, not an accuracy benchmark: it saves representative
annotated frames and per-class counts so a human can decide whether a second
model contributes useful detections before processing the full video.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def save_sheet(images: list[np.ndarray], path: Path, columns: int = 2) -> None:
    if not images:
        return
    height, width = images[0].shape[:2]
    rows = math.ceil(len(images) / columns)
    sheet = np.full((rows * height, columns * width, 3), 245, dtype=np.uint8)
    for index, image in enumerate(images):
        y, x = divmod(index, columns)
        sheet[y * height : (y + 1) * height, x * width : (x + 1) * width] = image
    cv2.imwrite(str(path), sheet)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--times", default="0,45,90,135,180,225,270,315")
    parser.add_argument("--imgsz", type=int, default=1280)
    args = parser.parse_args()

    from ultralytics import YOLO

    args.out.mkdir(parents=True, exist_ok=True)
    times = [float(value) for value in args.times.split(",")]
    cap = cv2.VideoCapture(str(args.clip))
    if not cap.isOpened():
        raise ValueError(f"Cannot open {args.clip}")
    frames = []
    for second in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
        ok, frame = cap.read()
        if ok:
            frames.append((second, frame))
    cap.release()

    models = {
        "broadcast": Path("models/best_broadcast.pt"),
        "topview": Path("models/best_topview.pt"),
    }
    audit = {"clip": str(args.clip), "imgsz": args.imgsz, "samples": []}
    for model_name, model_path in models.items():
        model = YOLO(model_path)
        previews = []
        for second, frame in frames:
            result = model.predict(frame, imgsz=args.imgsz, conf=0.10, verbose=False)[0]
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()
            names = result.names
            counts = Counter(str(names[int(class_id)]) for class_id in classes)
            audit["samples"].append(
                {
                    "model": model_name,
                    "time": second,
                    "total": len(boxes),
                    "classes": dict(counts),
                    "meanConfidence": float(confidences.mean()) if len(confidences) else None,
                }
            )
            preview = cv2.resize(frame, (960, 540))
            scale_x, scale_y = 960 / frame.shape[1], 540 / frame.shape[0]
            for box, class_id, confidence in zip(boxes, classes, confidences):
                x1, y1, x2, y2 = box
                color = (65, 190, 245) if "ball" in str(names[class_id]).lower() else (89, 207, 113)
                cv2.rectangle(preview, (round(x1 * scale_x), round(y1 * scale_y)), (round(x2 * scale_x), round(y2 * scale_y)), color, 1)
                if confidence >= 0.25:
                    cv2.putText(preview, str(names[class_id])[:3], (round(x1 * scale_x), max(12, round(y1 * scale_y) - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)
            label = f"{model_name} | {second:.0f}s | {len(boxes)} detections"
            cv2.rectangle(preview, (0, 0), (330, 29), (18, 25, 20), -1)
            cv2.putText(preview, label, (9, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(str(args.out / f"{model_name}-{round(second):03d}.jpg"), preview)
            previews.append(preview)
            (args.out / "model-domain-audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        save_sheet(previews, args.out / f"{model_name}-contact-sheet.jpg")
    print(args.out / "model-domain-audit.json")


if __name__ == "__main__":
    main()
