"""
Finds the least-camera-movement window in a clip, so you can trim a panning broadcast
clip down to a segment where a single frame-0 homography is actually valid, instead of
eyeballing the video for a "calm" stretch.

Why this exists: ViewTransformer computes ONE homography from frame 0's calibration and
reuses it for the whole clip. CameraMovementEstimator's optical-flow compensation only
corrects pixel-level TRANSLATION -- a real broadcast pan is usually the camera rotating
on its mount, which changes the actual pixel-to-pitch mapping, not just shifts it. So the
fix isn't to compensate harder, it's to calibrate on -- and only trust results from -- a
stretch of the clip where the camera barely moves at all.

This reuses the same motion signal CameraMovementEstimator uses (largest single tracked
background feature's frame-to-frame displacement, sampled from narrow strips along the
top/bottom of the frame so players are unlikely to be picked up), but streams frames
straight from disk via cv2.VideoCapture instead of loading the whole clip into memory
first -- the same OOM trap the notebook hit earlier in this project.

Usage:
    python find_static_segment.py --clip path\\to\\test.mp4 --window 6
    (prints the calmest 6-second window, plus a ready-to-run ffmpeg trim command)
"""

import argparse
import cv2
import numpy as np


FEATURE_PARAMS = dict(maxCorners=100, qualityLevel=0.3, minDistance=3, blockSize=7)
LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)


def _edge_mask(h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[0 : int(h * 0.08), :] = 1
    mask[int(h * 0.92) : h, :] = 1
    return mask


def per_frame_movement(clip_path, stride=1, max_dim=640):
    """Streams the clip frame by frame (never holds more than one frame in memory) and
    returns (native_fps, sampled_fps, movements) where movements[i] is the estimated
    camera-shift magnitude (pixels, at the analysis scale) between sampled frame i-1 and
    sampled frame i. movements[0] is always 0.0 (frame 0 has nothing to compare to)."""
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open clip: {clip_path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    movements = []
    old_gray = None
    old_features = None
    scale = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        if scale is None:
            h0, w0 = frame.shape[:2]
            scale = max_dim / max(h0, w0) if max(h0, w0) > max_dim else 1.0
        if scale != 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        mask = _edge_mask(h, w)

        if old_gray is None:
            movements.append(0.0)
            old_gray = gray
            old_features = cv2.goodFeaturesToTrack(gray, mask=mask, **FEATURE_PARAMS)
            frame_idx += 1
            continue

        if old_features is None or len(old_features) == 0:
            old_features = cv2.goodFeaturesToTrack(old_gray, mask=mask, **FEATURE_PARAMS)

        mag = 0.0
        if old_features is not None and len(old_features) > 0:
            new_features, status, _ = cv2.calcOpticalFlowPyrLK(old_gray, gray, old_features, None, **LK_PARAMS)
            if new_features is not None and status is not None:
                status = status.flatten()
                good_new = new_features[status == 1]
                good_old = old_features[status == 1]
                max_d = 0.0
                for new, old in zip(good_new, good_old):
                    d = float(np.hypot(*(new.ravel() - old.ravel())))
                    if d > max_d:
                        max_d = d
                mag = max_d
                old_features = good_new.reshape(-1, 1, 2) if len(good_new) > 0 else None

        if old_features is None or len(old_features) == 0:
            old_features = cv2.goodFeaturesToTrack(gray, mask=mask, **FEATURE_PARAMS)

        movements.append(mag)
        old_gray = gray
        frame_idx += 1

    cap.release()
    sampled_fps = native_fps / stride
    return native_fps, sampled_fps, movements


def best_window(movements, sampled_fps, window_seconds):
    window_frames = max(1, int(round(window_seconds * sampled_fps)))
    if window_frames >= len(movements):
        return 0, len(movements) - 1, float(sum(movements))

    # movements[i] is the shift arriving INTO sampled frame i, so the total motion
    # "inside" a window [start, end] is the sum of movements[start+1 .. end].
    prefix = [0.0]
    for m in movements:
        prefix.append(prefix[-1] + m)

    best_start, best_cost = 0, float("inf")
    for start in range(0, len(movements) - window_frames + 1):
        end = start + window_frames
        cost = prefix[end] - prefix[start + 1] if start + 1 <= end else prefix[end] - prefix[start]
        if cost < best_cost:
            best_cost = cost
            best_start = start
    return best_start, best_start + window_frames - 1, best_cost


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clip", required=True, help="Path to the broadcast clip")
    parser.add_argument("--window", type=float, default=6.0, help="Desired trimmed segment length, in seconds (default 6)")
    parser.add_argument("--stride", type=int, default=1, help="Analyze every Nth frame (default 1 = every frame; raise this for a very long clip to speed up scanning)")
    args = parser.parse_args()

    native_fps, sampled_fps, movements = per_frame_movement(args.clip, stride=args.stride)
    if len(movements) < 2:
        raise SystemExit("Clip too short to analyze.")

    start_idx, end_idx, cost = best_window(movements, sampled_fps, args.window)
    start_sec = (start_idx * args.stride) / native_fps
    end_sec = ((end_idx + 1) * args.stride) / native_fps
    avg_shift = cost / max(1, end_idx - start_idx)

    overall_avg = sum(movements) / max(1, len(movements) - 1)

    print(f"Clip: {args.clip}")
    print(f"Native fps: {native_fps:.2f}, analyzed {len(movements)} sampled frames (stride={args.stride})")
    print(f"Whole-clip average camera shift: {overall_avg:.2f}px/frame")
    print()
    print(f"Calmest {args.window:.1f}s window: {start_sec:.2f}s -> {end_sec:.2f}s")
    print(f"  average camera shift in this window: {avg_shift:.2f}px/frame (vs {overall_avg:.2f}px/frame overall)")
    print()
    print("Trim it with ffmpeg (re-encodes so the cut lands exactly on those timestamps):")
    print(f'  ffmpeg -i "{args.clip}" -ss {start_sec:.2f} -to {end_sec:.2f} -c:v libx264 -c:a aac "{args.clip.rsplit(".", 1)[0]}_trimmed.mp4"')
    print()
    print("Then re-run calibration on the TRIMMED clip, not the original:")
    print(f'  python run_local.py --clip "{args.clip.rsplit(".", 1)[0]}_trimmed.mp4" --mode calibrate')


if __name__ == "__main__":
    main()
