import sys
sys.path.append('/content')


class PlayerRatingModel:
    """
    Stage 4.2 -- a self-built, explainable 1-10 rating per player per match, from
    stats that are honestly measurable off single-camera 2D tracking: distance
    covered, sprint count, possession involvement, and ball recoveries (a
    proximity/possession-based turnover heuristic). Deliberately NOT presented as
    equivalent to FotMob/Sofascore's proprietary event-based ratings -- this is a
    transparent, from-scratch model built on a much smaller set of honestly-available
    inputs, and its weights are a stated, arguable design choice, not a hidden
    formula. Pass count (only meaningful with Stage 4.3's event detection) is
    intentionally left out here -- 4.3 wasn't built, so it's not silently assumed.

    Weighting rationale (why these weights, not others -- meant to be re-argued, not
    treated as settled):
      - distance_covered (25%): a durable, low-noise work-rate signal -- every
        player's distance is measured the same calibrated way (Stage 3.3's running
        total), so it anchors the volume side of the rating.
      - sprint_count (20%): distinguishes genuine high-intensity contribution from a
        player who covers the same ground at a jog -- frames above
        `sprint_speed_kmh` count.
      - possession_involvement (30%): the stat most tied to actually being part of
        the game rather than just running around it -- weighted highest of the four.
      - ball_recoveries (25%): a defensive/disruption signal the pure-movement stats
        above don't capture at all -- without it, a rating would silently reward only
        attacking-minded running and ignore defensive contribution entirely.
    A coach might reasonably reweight these per position (recoveries higher for a
    defensive mid, possession_involvement higher for a playmaker) -- this model uses
    one fixed set of weights for every player, which is itself a stated
    simplification worth naming, not hiding.
    """

    DEFAULT_WEIGHTS = {
        "distance_covered": 0.25,
        "sprint_count": 0.20,
        "possession_involvement": 0.30,
        "ball_recoveries": 0.25,
    }

    def __init__(self, weights=None, sprint_speed_kmh=20.0):
        self.weights = dict(weights) if weights is not None else dict(self.DEFAULT_WEIGHTS)
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        self.sprint_speed_kmh = sprint_speed_kmh

    def compute_raw_stats(self, tracks, team_ball_control):
        """
        tracks: pipeline tracks dict with `speed`, `distance`, `has_ball`, and `team`
        already populated per player per frame (Stages 2 and 3's own outputs).
        team_ball_control: per-frame sequence of which team (1/2, or 0) currently has
        the ball.

        Returns {track_id: {"team", "distance_covered", "sprint_count",
        "possession_involvement", "ball_recoveries"}}:
          - distance_covered: that player's own final cumulative `distance` (Stage
            3.3 already tracks a running total per player) -- the max value seen,
            since it's monotonically non-decreasing.
          - sprint_count: number of frames where that player's `speed` exceeded
            `sprint_speed_kmh`.
          - possession_involvement: number of frames flagged `has_ball` for that
            player (Stage 2.5's ball-assignment signal) -- proximity frames; a simple,
            honest proxy for "involved in the game," not a full pass/event count.
          - ball_recoveries: number of frames where team_ball_control changed TO that
            player's team on that exact frame AND that specific player is the one
            holding the ball that frame -- i.e. the player whose touch coincided with
            a turnover in their team's favor. A simple heuristic, explicitly not a
            modeled "tackle" or "interception" -- the same limitation the Build Plan
            itself names for this stat.
        """
        stats = {}

        def _ensure(track_id, team):
            if track_id not in stats:
                stats[track_id] = {
                    "team": team, "distance_covered": 0.0, "sprint_count": 0,
                    "possession_involvement": 0, "ball_recoveries": 0,
                }

        for frame_num, player_track in enumerate(tracks["players"]):
            prev_team = team_ball_control[frame_num - 1] if frame_num > 0 else 0
            cur_team = team_ball_control[frame_num] if frame_num < len(team_ball_control) else 0
            turnover_to = cur_team if (cur_team in (1, 2) and prev_team in (1, 2) and cur_team != prev_team) else None

            for track_id, track in player_track.items():
                team = track.get("team")
                if team not in (1, 2):
                    continue
                _ensure(track_id, team)

                dist = track.get("distance")
                if dist is not None:
                    stats[track_id]["distance_covered"] = max(stats[track_id]["distance_covered"], dist)

                speed = track.get("speed")
                if speed is not None and speed > self.sprint_speed_kmh and not track.get("motion_rejected"):
                    stats[track_id]["sprint_count"] += 1

                if track.get("has_ball"):
                    stats[track_id]["possession_involvement"] += 1
                    if turnover_to == team:
                        stats[track_id]["ball_recoveries"] += 1

        return stats

    @staticmethod
    def _normalize(values):
        """
        0-1 min-max normalizes a {id: value} dict. When every value is equal
        (including just one player) there's no spread to rank them apart by, so
        everyone normalizes to 1.0 rather than an arbitrary 0.0 -- a player isn't
        penalized just for being alone in, or tied within, the comparison set.
        """
        if not values:
            return {}
        lo, hi = min(values.values()), max(values.values())
        if hi - lo < 1e-9:
            return {k: 1.0 for k in values}
        return {k: (v - lo) / (hi - lo) for k, v in values.items()}

    def rate_players(self, tracks, team_ball_control):
        """
        Returns {track_id: {"team", "rating_1_10", "raw_stats", "normalized_stats"}}.
        Each stat is min-max normalized 0-1 ACROSS ALL PLAYERS IN THIS MATCH (per the
        Build Plan's own spec), weighted, summed, then mapped from [0,1] to [1,10].
        """
        raw_stats = self.compute_raw_stats(tracks, team_ball_control)

        normalized_by_stat = {}
        for stat_name in self.weights:
            values = {tid: s[stat_name] for tid, s in raw_stats.items()}
            normalized_by_stat[stat_name] = self._normalize(values)

        results = {}
        for track_id, raw in raw_stats.items():
            normalized = {stat: normalized_by_stat[stat][track_id] for stat in self.weights}
            score_0_1 = sum(self.weights[stat] * normalized[stat] for stat in self.weights)
            rating = 1.0 + score_0_1 * 9.0
            results[track_id] = {
                "team": raw["team"],
                "rating_1_10": round(rating, 2),
                "raw_stats": raw,
                "normalized_stats": {k: round(v, 3) for k, v in normalized.items()},
            }
        return results
