"""Build named player analytics and tactics from registered wide-camera tracks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from analysis import analyze
from ball_path import select_ball_path


EXPECTED = {
    1: {  # Liverpool, defending the right goal and attacking -x.
        "1": (100, 34), "66": (80, 10), "5": (82, 26), "4": (82, 42), "26": (80, 58),
        "3": (67, 34), "14": (65, 18), "6": (65, 50), "11": (54, 10), "10": (54, 34), "23": (54, 58),
    },
    2: {  # Real Madrid, defending the left goal and attacking +x.
        "1": (5, 34), "23": (25, 10), "4": (23, 26), "3": (23, 42), "2": (25, 58),
        "14": (38, 34), "8": (40, 18), "10": (40, 50), "20": (51, 10), "9": (51, 34), "15": (51, 58),
    },
}


def project(matrix: np.ndarray, x: float, y: float) -> list[float] | None:
    point = cv2.perspectiveTransform(np.float32([[[x, y]]]), matrix)[0, 0]
    if not np.isfinite(point).all() or not (-1 <= point[0] <= 106 and -1 <= point[1] <= 69):
        return None
    return [float(np.clip(point[0], 0, 105)), float(np.clip(point[1], 0, 68))]


def interpolate_ball(frames: list[dict], maximum_gap: float = 1.2) -> None:
    known = [index for index, frame in enumerate(frames) if frame["ball"] is not None]
    for left, right in zip(known, known[1:]):
        elapsed = frames[right]["time"] - frames[left]["time"]
        distance = np.linalg.norm(np.asarray(frames[right]["ball"]) - frames[left]["ball"])
        if right - left <= 1 or elapsed > maximum_gap or distance / elapsed > 40:
            continue
        for index in range(left + 1, right):
            ratio = (frames[index]["time"] - frames[left]["time"]) / elapsed
            frames[index]["ball"] = (
                np.asarray(frames[left]["ball"]) * (1 - ratio) + np.asarray(frames[right]["ball"]) * ratio
            ).round(3).tolist()
            frames[index]["ballSource"] = "bounded interpolation"


def roster_states(lineup: dict) -> dict[int, dict[str, dict]]:
    states = {1: {}, 2: {}}
    for team_id, team_name in ((1, "Liverpool"), (2, "Real Madrid")):
        for player in lineup["lineups"][team_name]:
            number = str(player["number"])
            states[team_id][number] = {
                **player,
                "id": f"{'LIV' if team_id == 1 else 'RMA'}-{number}",
                "team": team_id,
                "last": np.asarray(EXPECTED[team_id][number], dtype=float),
                "velocity": np.zeros(2),
                "time": None,
                "source": None,
            }
    return states


def assign_identities(frames: list[dict], lineup: dict) -> dict:
    states = roster_states(lineup)
    identity_meta = {}
    for team in states.values():
        for state in team.values():
            identity_meta[state["id"]] = {key: state[key] for key in ("name", "number", "position", "team")}
    seeded = {1: False, 2: False}
    for frame in frames:
        output = []
        for team_id in (1, 2):
            observations = [player for player in frame["players"] if player["team"] == team_id]
            identities = list(states[team_id].values())
            if not observations:
                continue
            costs = np.full((len(identities), len(observations)), 1e5, dtype=float)
            for row, identity in enumerate(identities):
                if identity["time"] is None:
                    prediction = identity["last"]
                    limit = 18.0 if not seeded[team_id] else 8.0
                else:
                    elapsed = frame["time"] - identity["time"]
                    prediction = identity["last"] + identity["velocity"] * min(elapsed, 1.0)
                    limit = min(14.0, 2.5 + 6.0 * elapsed)
                for column, observation in enumerate(observations):
                    is_goalkeeper = identity["position"] == "GK"
                    is_keeper_detection = observation.get("role") == "goalkeeper"
                    # A goalkeeper cannot inherit an outfield track after leaving
                    # the camera view. Conversely, detections explicitly labelled
                    # goalkeeper must not be given to an outfield identity.
                    if is_goalkeeper and not is_keeper_detection:
                        continue
                    if not is_goalkeeper and is_keeper_detection:
                        continue
                    distance = np.linalg.norm(np.asarray([observation["x"], observation["y"]]) - prediction)
                    continuity = -2.0 if observation["sourceId"] == identity["source"] else 0.0
                    keeper_centrality = abs(observation["y"] - 34) * 0.12 if is_goalkeeper else 0.0
                    if distance <= limit:
                        costs[row, column] = distance + continuity + keeper_centrality
            rows, columns = linear_sum_assignment(costs)
            used = set()
            for row, column in zip(rows, columns):
                if costs[row, column] >= 1e4:
                    continue
                identity, observation = identities[row], observations[column]
                position = np.asarray([observation["x"], observation["y"]])
                if identity["time"] is not None:
                    elapsed = frame["time"] - identity["time"]
                    velocity = (position - identity["last"]) / elapsed
                    if np.linalg.norm(velocity) <= 12.5:
                        identity["velocity"] = 0.55 * identity["velocity"] + 0.45 * velocity
                identity["last"], identity["time"], identity["source"] = position, frame["time"], observation["sourceId"]
                role = "goalkeeper" if identity["position"] == "GK" else "outfield"
                output.append({**observation, **identity_meta[identity["id"]], "id": identity["id"], "role": role, "observationSource": "detected"})
                used.add(column)
            if len(used) >= 7:
                seeded[team_id] = True
        frame["players"] = output
    return identity_meta


def interpolate_player_gaps(frames: list[dict], maximum_gap: float = 1.2) -> None:
    """Fill only short gaps bounded by detections of the same seeded identity."""
    observations: dict[str, list[tuple[int, dict]]] = {}
    for index, frame in enumerate(frames):
        for player in frame["players"]:
            observations.setdefault(player["id"], []).append((index, player))
    for records in observations.values():
        for (left_index, left), (right_index, right) in zip(records, records[1:]):
            elapsed = frames[right_index]["time"] - frames[left_index]["time"]
            if right_index - left_index <= 1 or elapsed > maximum_gap:
                continue
            displacement = math.hypot(right["x"] - left["x"], right["y"] - left["y"])
            if displacement / elapsed > 12.5:
                continue
            for index in range(left_index + 1, right_index):
                ratio = (frames[index]["time"] - frames[left_index]["time"]) / elapsed
                frames[index]["players"].append({
                    **left,
                    "sourceId": f'interpolated:{left["id"]}',
                    "x": round(left["x"] * (1 - ratio) + right["x"] * ratio, 3),
                    "y": round(left["y"] * (1 - ratio) + right["y"] * ratio, 3),
                    "observationSource": "bounded interpolation",
                })


def add_velocities(frames: list[dict]) -> None:
    history: dict[str, list[tuple[float, np.ndarray]]] = {}
    for frame in frames:
        for player in frame["players"]:
            records = history.setdefault(player["id"], [])
            prior = next((item for item in reversed(records) if 0.8 <= frame["time"] - item[0] <= 1.4), None)
            if prior:
                elapsed = frame["time"] - prior[0]
                velocity = (np.asarray([player["x"], player["y"]]) - prior[1]) / elapsed
                if np.linalg.norm(velocity) <= 12.5:
                    player["vx"], player["vy"] = round(float(velocity[0]), 3), round(float(velocity[1]), 3)
                else:
                    player["vx"] = player["vy"] = None
            else:
                player["vx"] = player["vy"] = None
            records.append((frame["time"], np.asarray([player["x"], player["y"]])))
            if len(records) > 12:
                records.pop(0)


def normalize(values: dict[str, float]) -> dict[str, float]:
    minimum, maximum = min(values.values(), default=0), max(values.values(), default=0)
    return {key: 0.5 if maximum == minimum else (value - minimum) / (maximum - minimum) for key, value in values.items()}


def analytics(payload: dict, identity_meta: dict) -> None:
    interval = payload["stride"] / payload["fps"]
    stats = {
        player_id: {"observedSeconds": 0.0, "distanceM": 0.0, "highSpeedSeconds": 0.0, "peakSpeedKmh": 0.0, "ballProximitySeconds": 0.0, "pressures": 0, "recoveries": 0, "positions": []}
        for player_id in identity_meta
    }
    previous = {}
    previous_pressers: set[str] = set()
    for frame in payload["frames"]:
        pressers = {item["id"] for shape in frame["shapes"].values() for item in shape.get("pressers", [])}
        for player in frame["players"]:
            item = stats[player["id"]]
            item["observedSeconds"] += interval
            item["positions"].append([player["x"], player["y"]])
            before = previous.get(player["id"])
            if before and frame["time"] - before[0] <= interval * 1.6:
                distance = math.hypot(player["x"] - before[1], player["y"] - before[2])
                speed = distance / (frame["time"] - before[0]) * 3.6
                if speed <= 38:
                    item["distanceM"] += distance
                    item["peakSpeedKmh"] = max(item["peakSpeedKmh"], speed)
                    if speed >= 19.8:
                        item["highSpeedSeconds"] += interval
            if frame["ball"] and math.hypot(player["x"] - frame["ball"][0], player["y"] - frame["ball"][1]) <= 2.2:
                item["ballProximitySeconds"] += interval
            if player["id"] in pressers and player["id"] not in previous_pressers:
                item["pressures"] += 1
            previous[player["id"]] = (frame["time"], player["x"], player["y"])
        previous_pressers = pressers
    for event in payload["events"]:
        if event["type"] != "turnover":
            continue
        frame = payload["frames"][event["frameIndex"]]
        candidates = [player for player in frame["players"] if player["team"] == event["team"] and frame["ball"]]
        if candidates:
            closest = min(candidates, key=lambda player: math.hypot(player["x"] - frame["ball"][0], player["y"] - frame["ball"][1]))
            stats[closest["id"]]["recoveries"] += 1
    factors = {
        "distanceM": normalize({key: value["distanceM"] for key, value in stats.items()}),
        "highSpeedSeconds": normalize({key: value["highSpeedSeconds"] for key, value in stats.items()}),
        "ballProximitySeconds": normalize({key: value["ballProximitySeconds"] for key, value in stats.items()}),
        "pressures": normalize({key: value["pressures"] for key, value in stats.items()}),
        "recoveries": normalize({key: value["recoveries"] for key, value in stats.items()}),
    }
    weights = {"distanceM": 0.25, "highSpeedSeconds": 0.15, "ballProximitySeconds": 0.25, "pressures": 0.20, "recoveries": 0.15}
    ratings = {}
    for player_id, values in stats.items():
        score = sum(weights[key] * factors[key][player_id] for key in weights)
        visible_ratio = values["observedSeconds"] / payload["duration"]
        rating = 5.5 + 3.0 * score if visible_ratio >= 0.20 else None
        ratings[player_id] = {
            **identity_meta[player_id],
            **{key: round(value, 2) if isinstance(value, float) else value for key, value in values.items() if key != "positions"},
            "rating": round(rating, 1) if rating is not None else None,
            "visibleRatio": round(visible_ratio, 3),
        }
    payload["ratings"] = ratings
    payload["ratingModel"] = {
        "name": "five-minute visible activity rating",
        "range": "5.5-8.5",
        "weights": weights,
        "minimumVisibility": 0.20,
        "limitations": "Relative activity score for this clip. It is not a full-match performance rating and does not infer goals, assists, passes, shots, or xG.",
    }
    payload["heatmaps"] = {player_id: values["positions"] for player_id, values in stats.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--lineup", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    tracked = json.loads((args.run / "pixel-tracks.json").read_text(encoding="utf-8"))
    registration = json.loads((args.run / "registrations.json").read_text(encoding="utf-8"))
    lineup = json.loads(args.lineup.read_text(encoding="utf-8"))
    matrices = {record["frame"]: np.asarray(record["imageToPitch"], dtype=float) for record in registration["frames"]}
    frames = []
    for source in tracked["frames"]:
        matrix = matrices.get(source["frame"])
        if matrix is None:
            continue
        players = []
        for player in source["players"]:
            box = player["bbox"]
            position = project(matrix, (box[0] + box[2]) / 2, box[3])
            if position is None:
                continue
            team = player["team"]
            side_assigned_keeper = False
            # Kit color is unreliable for goalkeepers. The supplied attack
            # directions tell us which goal each side is defending, so an
            # otherwise-unassigned detection in either six-yard area can be
            # assigned without relying on shirt color or a model class.
            if team is None:
                team = 1 if position[0] >= 93 else 2 if position[0] <= 12 else None
                side_assigned_keeper = team is not None
            if team not in (1, 2):
                continue
            players.append({
                "id": player["id"], "sourceId": player["id"], "team": team,
                "teamConfidence": player["teamConfidence"], "x": round(position[0], 3), "y": round(position[1], 3),
                "role": "goalkeeper" if player["detectorClass"] == "goalkeeper" or side_assigned_keeper else "outfield",
            })
        candidates = []
        for ball in source["ballCandidates"]:
            box = ball["bbox"]
            position = project(matrix, (box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            if position is not None:
                nearest = min(
                    (math.hypot(player["x"] - position[0], player["y"] - position[1]) for player in players),
                    default=25.0,
                )
                candidates.append({
                    "position": position,
                    "confidence": ball["confidence"],
                    "nearestPlayerDistance": round(nearest, 3),
                    "method": "broadcast model",
                })
        frames.append({"frame": source["frame"], "time": source["time"], "players": players, "candidates": candidates})
    select_ball_path(frames)
    interpolate_ball(frames)
    identity_meta = assign_identities(frames, lineup)
    interpolate_player_gaps(frames)
    add_velocities(frames)
    payload = {
        "schemaVersion": 2,
        "clip": tracked["clip"],
        "fixture": lineup["fixture"],
        "competition": lineup["competition"],
        "fps": tracked["fps"],
        "stride": tracked["stride"],
        "duration": tracked["frameCount"] / tracked["fps"],
        "pitch": {"length": 105, "width": 68, "dimensionsVerified": False},
        "teams": [
            {"id": 1, "name": "Liverpool", "shortName": "LIV", "color": "#c83b32", "attacks": -1},
            {"id": 2, "name": "Real Madrid", "shortName": "RMA", "color": "#f2efe6", "attacks": 1},
        ],
        "teamAssignments": identity_meta,
        "teamMethod": tracked["teamMethod"],
        "identityMethod": "Known lineup seeded from the supplied mirrored 4-3-3, followed by motion continuity. Short gaps up to 1.2 seconds are interpolated; names remain provisional until jersey-number anchors are reviewed.",
        "frames": frames,
        "source": "dual-model wide broadcast pipeline",
        "calibration": {
            "method": registration["method"],
            "failedSteps": registration["failedSteps"],
            "dimensions": "assumed 105 x 68 m",
            "referenceFrameReviewed": True,
        },
    }
    payload = analyze(payload)
    analytics(payload, identity_meta)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "tactical.json").write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    summary = {
        **payload["summary"],
        "namedPlayerObservations": sum(len(frame["players"]) for frame in frames),
        "playersWithRatings": sum(value["rating"] is not None for value in payload["ratings"].values()),
        "registrationFailures": registration["failedSteps"],
    }
    (args.out / "build-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
