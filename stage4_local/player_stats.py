"""Clip statistics derived from observed tracks, with explicit measurement coverage."""
import numpy as np


def build_player_stats(tracks, fps):
    samples = {}
    for fn, frame in enumerate(tracks["players"]):
        for tid, player in frame.items():
            samples.setdefault(str(tid), []).append((fn, player))
    results = {}
    for tid, observed in samples.items():
        positions = [(fn, p) for fn, p in observed if p.get("position_transformed") is not None]
        speeds = [float(p["speed"]) for _, p in observed if p.get("speed") is not None and not p.get("motion_rejected")]
        controlled = [fn for fn, p in observed if p.get("has_ball")]
        bouts = sum(i == 0 or fn != controlled[i-1]+1 for i, fn in enumerate(controlled))
        sprint_frames = [fn for fn, p in observed if p.get("speed", 0) > 20 and not p.get("motion_rejected")]
        results[tid] = {
            "observed_frames": len(observed), "observed_seconds": len(observed)/fps,
            "measured_frames": len(positions), "measured_seconds": len(positions)/fps,
            "measured_coverage": len(positions)/len(observed),
            "speed_samples": len(speeds),
            "average_speed_kmh": float(np.mean(speeds)) if speeds else None,
            "peak_speed_kmh": max(speeds) if speeds else None,
            "sprint_seconds": len(sprint_frames)/fps,
            "possession_seconds": len(controlled)/fps,
            "possession_sequences": bouts,
            "rejected_motion_frames": sum(bool(p.get("motion_rejected")) for _,p in observed),
            "first_seen_seconds": observed[0][0]/fps,
            "last_seen_seconds": observed[-1][0]/fps,
        }
    return results
