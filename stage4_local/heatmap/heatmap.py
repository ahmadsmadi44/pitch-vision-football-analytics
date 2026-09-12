import sys
sys.path.append('/content')

import numpy as np


class PositionHeatmapBuilder:
    """
    Stage 4.1 -- accumulates each player's real-world (meters) position across every
    frame they were tracked, and bins it into a 2D density grid over the pitch --
    "where did this player spend the match." Exported as a JSON-serializable grid
    (raw counts, and a normalized density that sums to 1 over sampled cells), NOT a
    baked image -- a future frontend can render it however it wants (color scale,
    opacity, zoom) instead of being stuck with a fixed picture.

    Deliberately a plain 2D histogram rather than a full kernel-density estimate --
    coarser, but exactly reproducible and trivially testable; a smoother KDE-style
    look is a rendering choice a frontend can add on top of this raw grid (e.g. a
    blur pass) if it wants one, without changing what's actually being measured here.
    """

    def __init__(self, pitch_length_m, pitch_width_m, cell_size_m=2.0):
        self.pitch_length_m = pitch_length_m
        self.pitch_width_m = pitch_width_m
        self.cell_size_m = cell_size_m
        self.n_cols = int(np.ceil(pitch_length_m / cell_size_m))
        self.n_rows = int(np.ceil(pitch_width_m / cell_size_m))

    def _cell_index(self, x, y):
        col = int(x // self.cell_size_m)
        row = int(y // self.cell_size_m)
        col = max(0, min(self.n_cols - 1, col))
        row = max(0, min(self.n_rows - 1, row))
        return row, col

    def build_player_grids(self, tracks):
        """
        tracks: pipeline tracks dict, with `position_transformed` already populated
        by ViewTransformer.transform_tracks().

        Returns {track_id: {"team": 1/2/"referee", "grid": [[raw counts]],
        "n_samples": n}} -- one entry per player who had at least one valid
        real-world position anywhere in the clip. `grid` is n_rows x n_cols of RAW
        counts (not yet normalized), so grids can still be combined/compared before
        normalizing.
        """
        grids = {}
        for player_track in tracks["players"]:
            for track_id, track in player_track.items():
                pos = track.get("position_transformed")
                if pos is None:
                    continue
                if track_id not in grids:
                    grids[track_id] = {
                        "team": track.get("team"),
                        "grid": [[0] * self.n_cols for _ in range(self.n_rows)],
                        "n_samples": 0,
                    }
                row, col = self._cell_index(pos[0], pos[1])
                grids[track_id]["grid"][row][col] += 1
                grids[track_id]["n_samples"] += 1
        return grids

    def to_density_json(self, grids):
        """
        Normalizes each player's raw-count grid to a density (sums to 1 across all
        cells) for JSON export. A player with 0 samples keeps an all-zero grid rather
        than dividing by zero.
        """
        result = {}
        for track_id, entry in grids.items():
            n = entry["n_samples"]
            if n > 0:
                density = [[c / n for c in row] for row in entry["grid"]]
            else:
                density = entry["grid"]
            result[str(track_id)] = {
                "team": entry["team"],
                "density_grid": density,
                "n_samples": n,
                "cell_size_m": self.cell_size_m,
                "pitch_length_m": self.pitch_length_m,
                "pitch_width_m": self.pitch_width_m,
            }
        return result
