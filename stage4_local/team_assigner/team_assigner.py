import cv2
import numpy as np
from sklearn.cluster import KMeans


class TeamAssigner:
    """
    Assigns each tracked person to a team by clustering shirt color, and — since our
    detector only has a 'player' class (no separate 'referee' class in training data) —
    flags anyone whose shirt color doesn't cleanly match either team cluster as a
    non-player (referee) instead of forcing them into the nearest team.

    History of what didn't work, kept here because the next person touching this file
    (possibly future-me) will otherwise re-try the same dead ends:
      1. Top-half-of-box sampling (broadcast/side-view assumption: shirt on top, shorts
         below) -- wrong for an overhead view, where the top of a tiny box is head/hair.
      2. Full-box + inner 2-cluster KMeans, treating corner pixels as "background" --
         fails on tight boxes where the corners are still the player's own body.
      3. Plain median of the whole box, even the whole native-resolution box -- still
         failed in practice. Measured real output looked like BGR (91, 133, 109) and
         (101, 145, 124) for the two teams: nearly identical AND both green-dominant
         (G channel highest in both) -- a dead giveaway that grass pixels, not shirt
         pixels, were winning the median vote. A generously-sized/loosely-fit detection
         box around a small player can be majority background even when "tight" by eye.
    """

    # Grass in this footage (real turf, mowing stripes) sits in a fairly consistent
    # green hue band regardless of light/dark stripe -- stripes differ in brightness
    # (V), not hue. OpenCV hue is 0-179. Saturation gate avoids excluding dark/desaturated
    # shirts that merely happen to fall in the same hue range.
    GRASS_HUE_LOW = 25
    GRASS_HUE_HIGH = 95
    GRASS_SAT_MIN = 40

    def __init__(self, view="topview"):
        if view not in ("topview", "broadcast"):
            raise ValueError("view must be topview or broadcast")
        self.view = view
        self.team_colors = {}
        self.player_team_dict = {}
        self.kmeans = None
        self.outlier_threshold = None

    def get_player_color(self, frame, bbox):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h_frame, w_frame = frame.shape[:2]
        x1, x2 = max(0, x1), min(w_frame, x2)
        y1, y2 = max(0, y1), min(h_frame, y2)
        image = frame[y1:y2, x1:x2]
        if image.size == 0:
            return np.array([0, 0, 0])

        # Broadcast boxes include black shorts, bare legs, and socks. Their median
        # can be far from a white shirt even after grass removal. An overhead crop
        # has different geometry, so keep its established full-body sampling.
        if self.view == "broadcast":
            height = image.shape[0]
            torso = image[int(height * 0.15):max(int(height * 0.55), 1)]
            if torso.size:
                image = torso

        # Trim a small margin off each edge -- the outermost pixels of even a tight box
        # are the most likely to be anti-aliased/motion-blurred blends with whatever's
        # just outside the player.
        h, w = image.shape[:2]
        my, mx = int(h * 0.1), int(w * 0.1)
        if h - 2 * my > 0 and w - 2 * mx > 0:
            image = image[my:h - my, mx:w - mx]

        # Explicitly drop grass-hued pixels before summarizing color. This is a more
        # direct fix than just hoping a tighter crop avoids background: it targets the
        # actual contamination (green pitch) by color, so it still works even when the
        # detection box itself is loose or the player is tiny and blurry.
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue, sat = hsv[:, :, 0], hsv[:, :, 1]
        is_grass = (hue >= self.GRASS_HUE_LOW) & (hue <= self.GRASS_HUE_HIGH) & (sat >= self.GRASS_SAT_MIN)

        pixels = image.reshape(-1, 3)
        keep = ~is_grass.reshape(-1)
        kept_pixels = pixels[keep]

        # If almost everything got excluded (box is nearly all pitch -- heavy occlusion,
        # a bad box, or a genuinely green/olive kit), fall back to the full crop rather
        # than return a median of a handful of pixels.
        if len(kept_pixels) < 0.15 * len(pixels):
            kept_pixels = pixels

        return np.median(kept_pixels, axis=0)

    def assign_team_colors_from_samples(self, player_colors_dict):
        """
        Lower-level entry point: takes a pre-computed {track_id: color} mapping (e.g.
        each player's color already aggregated/medianed across several frames by the
        caller) and does the actual team clustering + referee-outlier detection.
        Separated from assign_team_colors() so the pipeline can aggregate samples across
        multiple frames for stability instead of trusting a single frame.
        """
        track_ids = list(player_colors_dict.keys())
        player_colors = np.array([player_colors_dict[tid] for tid in track_ids])

        if len(player_colors) < 2:
            raise ValueError("Need at least two sampled tracks to infer two team colors")
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=0)
        kmeans.fit(player_colors)
        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

        # distance from each person's color to their nearest team-cluster center — a
        # referee's kit color won't match either team well, so this distance spikes for
        # them specifically. Threshold is data-driven (mean + 2*std), not a hardcoded guess.
        distances = []
        for color in player_colors:
            label = kmeans.predict(color.reshape(1, -1))[0]
            distances.append(np.linalg.norm(color - kmeans.cluster_centers_[label]))
        distances = np.array(distances)
        self.outlier_threshold = distances.mean() + 2 * distances.std() if len(distances) > 1 else np.inf
        # On well-separated kits, a tight cluster makes mean+2*std reject a
        # brighter version of the same white shirt. Reserve a small fraction of
        # the between-kit separation for lighting variation. This remains an
        # appearance heuristic, not a reliable referee detector.
        separation = np.linalg.norm(kmeans.cluster_centers_[0] - kmeans.cluster_centers_[1])
        self.outlier_threshold = max(self.outlier_threshold, .25 * separation)

        for track_id, color in zip(track_ids, player_colors):
            label = kmeans.predict(color.reshape(1, -1))[0]
            dist = np.linalg.norm(color - kmeans.cluster_centers_[label])
            self.player_team_dict[track_id] = "referee" if dist > self.outlier_threshold else int(label) + 1

    def assign_team_colors(self, frame, player_detections):
        """Single-frame convenience wrapper around assign_team_colors_from_samples()."""
        player_colors = {
            track_id: self.get_player_color(frame, detection["bbox"])
            for track_id, detection in player_detections.items()
        }
        self.assign_team_colors_from_samples(player_colors)

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        color = self.get_player_color(frame, player_bbox)
        label = self.kmeans.predict(color.reshape(1, -1))[0]
        dist = np.linalg.norm(color - self.kmeans.cluster_centers_[label])

        team = "referee" if (self.outlier_threshold is not None and dist > self.outlier_threshold) else int(label) + 1
        self.player_team_dict[player_id] = team
        return team
