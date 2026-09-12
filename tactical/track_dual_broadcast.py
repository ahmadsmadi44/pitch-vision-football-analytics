"""Track checkpointed detections and assign Liverpool/Real Madrid kit teams."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from detect_dual_broadcast import iou


def classify_red_white(samples: dict[str, list[list[float]]], stats: dict[str, dict]) -> tuple[dict, dict]:
    all_features = {
        track_id: np.median(np.asarray(values, dtype=float), axis=0)
        for track_id, values in samples.items()
        if len(values) >= 2 and not stats[track_id]["goalkeeper"]
    }
    features = {track_id: value for track_id, value in all_features.items() if stats[track_id]["frames"] >= 5}
    keys = list(features)
    if len(keys) < 4:
        return {}, {"method": "insufficient appearance tracks"}
    matrix = np.asarray([features[key] for key in keys])
    scale = np.maximum(np.std(matrix, axis=0), 0.06)
    normalized = matrix / scale
    model = KMeans(n_clusters=2, n_init=30, random_state=22).fit(normalized)
    centres = model.cluster_centers_ * scale
    red_cluster = int(np.argmax(centres[:, -2]))
    white_cluster = int(np.argmax(centres[:, -4]))
    if red_cluster == white_cluster:
        white_cluster = 1 - red_cluster
    assignments = {}
    for track_id, point in zip(keys, normalized):
        distances = model.transform([point])[0]
        cluster = int(np.argmin(distances))
        margin = float((max(distances) - min(distances)) / max(max(distances), 1e-6))
        votes = [int(model.predict([np.asarray(value) / scale])[0]) == cluster for value in samples[track_id]]
        agreement = float(np.mean(votes))
        confidence = margin * agreement
        team = 1 if cluster == red_cluster else 2 if cluster == white_cluster else None
        assignments[track_id] = {
            "team": team if confidence >= 0.30 else None,
            "confidence": round(confidence, 3),
            "samples": len(samples[track_id]),
            "agreement": round(agreement, 3),
            "method": "temporal red/white appearance clustering",
        }
    # These kits are exceptionally separable. Direct temporal colour evidence
    # fills short fragments without forcing dark/ambiguous crops into a team.
    for track_id, point in all_features.items():
        white_fraction, red_fraction = float(point[-4]), float(point[-2])
        if red_fraction >= 0.12 and red_fraction > 1.5 * white_fraction:
            assignments[track_id] = {
                "team": 1, "confidence": round(min(1.0, red_fraction / 0.35), 3),
                "samples": len(samples[track_id]), "agreement": None,
                "method": "direct temporal red-kit evidence",
            }
        elif white_fraction >= 0.35 and white_fraction > 1.5 * red_fraction:
            assignments[track_id] = {
                "team": 2, "confidence": round(min(1.0, white_fraction / 0.80), 3),
                "samples": len(samples[track_id]), "agreement": None,
                "method": "direct temporal white-kit evidence",
            }
    return assignments, {
        "method": "native non-grass Lab appearance, track median, two-cluster temporal agreement",
        "semanticLabels": {"1": "Liverpool red", "2": "Real Madrid white"},
        "minimumConfidence": 0.30,
        "eligibleTracks": len(keys),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()

    import supervision as sv

    meta = json.loads((args.run / "detection-meta.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (args.run / "detections.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records.sort(key=lambda record: record["frame"])
    tracker = sv.ByteTrack(
        frame_rate=round(meta["fps"] / meta["stride"]),
        lost_track_buffer=50,
        track_activation_threshold=0.20,
        minimum_matching_threshold=0.72,
    )
    samples: dict[str, list[list[float]]] = defaultdict(list)
    stats = defaultdict(lambda: {"frames": 0, "sources": Counter(), "confidence": [], "goalkeeper": False})
    frames = []
    for record in records:
        detections = record["players"]
        if detections:
            boxes = np.asarray([item["bbox"] for item in detections], dtype=np.float32)
            confidences = np.asarray([item["confidence"] for item in detections], dtype=np.float32)
            tracked = tracker.update_with_detections(
                sv.Detections(
                    xyxy=boxes,
                    confidence=confidences,
                    class_id=np.zeros(len(boxes), dtype=int),
                )
            )
        else:
            boxes = np.empty((0, 4), dtype=np.float32)
            tracked = tracker.update_with_detections(sv.Detections.empty())
        players = []
        for tracked_box, track_id, track_confidence in zip(tracked.xyxy, tracked.tracker_id, tracked.confidence):
            if len(boxes) == 0:
                continue
            overlaps = np.asarray([iou(tracked_box, box) for box in boxes])
            match = int(np.argmax(overlaps))
            if overlaps[match] < 0.25:
                centres = (boxes[:, :2] + boxes[:, 2:]) / 2
                centre = (tracked_box[:2] + tracked_box[2:]) / 2
                match = int(np.argmin(np.linalg.norm(centres - centre, axis=1)))
            source = detections[match]
            key = str(int(track_id))
            if source.get("appearance") is not None:
                samples[key].append(source["appearance"])
            stats[key]["frames"] += 1
            stats[key]["sources"][source["source"]] += 1
            stats[key]["confidence"].append(float(track_confidence))
            stats[key]["goalkeeper"] |= source["detectorClass"] == "goalkeeper"
            players.append(
                {
                    "id": key,
                    "bbox": [round(float(value), 3) for value in tracked_box],
                    "confidence": round(float(track_confidence), 4),
                    "source": source["source"],
                    "detectorClass": source["detectorClass"],
                }
            )
        frames.append(
            {
                "frame": record["frame"],
                "time": record["time"],
                "fieldCoverage": record["fieldCoverage"],
                "players": players,
                "ballCandidates": record["ballCandidates"],
            }
        )

    assignments, method = classify_red_white(samples, stats)
    track_stats = {}
    retained = set()
    for track_id, values in stats.items():
        topview_only = values["sources"]["broadcast"] == 0
        keep = values["frames"] >= (5 if topview_only else 3)
        if keep:
            retained.add(track_id)
        track_stats[track_id] = {
            "frames": values["frames"],
            "sources": dict(values["sources"]),
            "meanConfidence": round(float(np.mean(values["confidence"])), 3),
            "goalkeeperCandidate": values["goalkeeper"],
            "retained": keep,
            "appearanceMedian": np.median(np.asarray(samples[track_id]), axis=0).round(5).tolist() if samples.get(track_id) else None,
            **assignments.get(track_id, {"team": None, "confidence": 0, "method": "unassigned"}),
        }
    for frame in frames:
        frame["players"] = [
            {**player, "team": track_stats[player["id"]]["team"], "teamConfidence": track_stats[player["id"]]["confidence"]}
            for player in frame["players"]
            if player["id"] in retained
        ]

    output = {
        "schemaVersion": 1,
        "clip": meta["clip"],
        "fps": meta["fps"],
        "stride": meta["stride"],
        "frameCount": meta["frameCount"],
        "teamMethod": method,
        "trackStats": track_stats,
        "frames": frames,
    }
    (args.run / "pixel-tracks.json").write_text(json.dumps(output, separators=(",", ":")), encoding="utf-8")
    per_frame = [len(frame["players"]) for frame in frames]
    summary = {
        "sampledFrames": len(frames),
        "retainedTracks": len(retained),
        "meanPlayersPerFrame": round(float(np.mean(per_frame)), 2),
        "minimumPlayers": min(per_frame, default=0),
        "maximumPlayers": max(per_frame, default=0),
        "assignedTeamObservations": sum(player["team"] is not None for frame in frames for player in frame["players"]),
        "playerObservations": sum(per_frame),
    }
    (args.run / "tracking-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
