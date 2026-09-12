"""Track a moving broadcast camera and map each sampled frame to pitch metres."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


# Manually audited against frame zero. Extra centre-circle and penalty-area
# correspondences make this more stable than extrapolating from hidden corners.
REFERENCE_WORLD = np.float32(
    [
        [0, 0], [52.5, 0], [105, 0], [52.5, 68],
        [52.5, 24.85], [52.5, 43.15], [43.35, 34], [61.65, 34],
        [16.5, 13.84], [16.5, 54.16], [88.5, 13.84], [88.5, 54.16],
    ]
)
REFERENCE_IMAGE_1920 = np.float32(
    [
        [293, 420], [943, 422], [1572, 422], [944, 847],
        [943, 524], [943, 630], [801, 576], [1085, 576],
        [459, 479], [246, 723], [1426, 479], [1615, 708],
    ]
)


def green_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([28, 28, 28]), np.array([96, 255, 255]))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def incremental_registration(previous: np.ndarray, current: np.ndarray) -> tuple[np.ndarray | None, dict]:
    previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    mask = green_mask(previous)
    points = cv2.goodFeaturesToTrack(previous_gray, maxCorners=1400, qualityLevel=0.008, minDistance=7, mask=mask, blockSize=5)
    if points is None or len(points) < 30:
        return None, {"features": 0, "inliers": 0}
    moved, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        points,
        None,
        winSize=(31, 31),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 35, 0.01),
    )
    valid = status.ravel().astype(bool)
    before, after = points[valid, 0], moved[valid, 0]
    inside = green_mask(current)[np.clip(after[:, 1].astype(int), 0, current.shape[0] - 1), np.clip(after[:, 0].astype(int), 0, current.shape[1] - 1)] > 0
    before, after = before[inside], after[inside]
    if len(before) < 24:
        return None, {"features": int(len(before)), "inliers": 0}
    current_to_previous, inlier_mask = cv2.findHomography(after, before, cv2.RANSAC, 1.8)
    inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    if current_to_previous is None or inliers < 20 or inliers / len(before) < 0.45:
        return None, {"features": int(len(before)), "inliers": inliers}
    current_to_previous /= current_to_previous[2, 2]
    return current_to_previous, {"features": int(len(before)), "inliers": inliers}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()

    meta = json.loads((args.run / "detection-meta.json").read_text(encoding="utf-8"))
    cap = cv2.VideoCapture(str(args.clip))
    if not cap.isOpened():
        raise ValueError(f"Cannot open {args.clip}")
    scale = np.float32([meta["width"] / 1920, meta["height"] / 1080])
    reference_image = REFERENCE_IMAGE_1920 * scale
    reference_to_world, _ = cv2.findHomography(reference_image, REFERENCE_WORLD, 0)
    previous_small = None
    cumulative_to_reference = np.eye(3)
    records = []
    failures = 0
    for frame_index in range(meta["frameCount"]):
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % meta["stride"]:
            continue
        small = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
        if previous_small is None:
            diagnostics = {"features": 0, "inliers": 0}
            valid = True
        else:
            step, diagnostics = incremental_registration(previous_small, small)
            valid = step is not None
            if valid:
                # Optical flow was measured at half resolution. Lift the transform
                # into native image coordinates before chaining it.
                down = np.diag([0.5, 0.5, 1.0])
                step_native = np.linalg.inv(down) @ step @ down
                cumulative_to_reference = cumulative_to_reference @ step_native
                cumulative_to_reference /= cumulative_to_reference[2, 2]
            else:
                failures += 1
        image_to_world = reference_to_world @ cumulative_to_reference
        image_to_world /= image_to_world[2, 2]
        records.append(
            {
                "frame": frame_index,
                "time": round(frame_index / meta["fps"], 6),
                "imageToPitch": image_to_world.tolist(),
                "stepValid": valid,
                **diagnostics,
            }
        )
        if valid or previous_small is None:
            previous_small = small
        if len(records) % 100 == 0:
            print(f"Registered {len(records)} frames; {failures} failed steps", flush=True)
    cap.release()
    output = {
        "schemaVersion": 1,
        "clip": str(args.clip),
        "fps": meta["fps"],
        "stride": meta["stride"],
        "method": "manual frame-zero pitch landmarks plus chained pitch-masked Lucas-Kanade homographies",
        "referenceWorld": REFERENCE_WORLD.tolist(),
        "referenceImage": reference_image.tolist(),
        "failedSteps": failures,
        "frames": records,
    }
    (args.run / "registrations.json").write_text(json.dumps(output, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({"registeredFrames": len(records), "failedSteps": failures}, indent=2))


if __name__ == "__main__":
    main()
