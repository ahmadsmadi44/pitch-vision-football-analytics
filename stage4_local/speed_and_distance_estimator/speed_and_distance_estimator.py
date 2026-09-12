import sys
sys.path.append('/content')

import cv2
import numpy as np

from utils.bbox_utils import measure_distance, get_foot_position


class SpeedAndDistanceEstimator:
    """
    Turns each player's real-world (meters) position over time into running distance
    covered (meters) and instantaneous speed (km/h). Computed over a rolling window of
    several frames rather than frame-to-frame -- frame-to-frame position jitter (a few
    pixels of detection noise, which becomes a few centimeters of real-world noise)
    would otherwise translate into wildly unstable speed readings.
    """

    def __init__(self, frame_window=5, fps=25, max_speed_kmh=45):
        self.frame_window = frame_window
        self.fps = fps
        self.max_speed_kmh = max_speed_kmh
        self.rejected_windows = 0

    def add_speed_and_distance(self, tracks):
        if self.fps <= 0 or self.frame_window < 1:
            raise ValueError("FPS and measurement window must be positive")
        for frame in tracks["players"]:
            for player in frame.values():
                for key in ("speed", "distance", "motion_rejected"):
                    player.pop(key, None)
        total_distance = {}
        self.rejected_windows = 0
        n_frames = len(tracks["players"])
        # A short median filter suppresses single-frame bbox jitter. Do not bridge
        # occlusion or uncalibrated frames: those contain no measured movement.
        smoothed = {}
        for fn, frame in enumerate(tracks["players"]):
            for tid, player in frame.items():
                if player.get("position_transformed") is None:
                    continue
                points = []
                for index in range(max(0, fn - 2), min(n_frames, fn + 3)):
                    position = tracks["players"][index].get(tid, {}).get("position_transformed")
                    if position is not None:
                        points.append(position)
                smoothed[fn, tid] = np.median(points, axis=0).tolist()

        for frame_num in range(0, n_frames, self.frame_window):
            last_frame = min(frame_num + self.frame_window, n_frames - 1)
            if last_frame == frame_num:
                continue

            for track_id, track in tracks["players"][frame_num].items():
                if any((fn, track_id) not in smoothed for fn in range(frame_num, last_frame + 1)):
                    continue

                start_pos = smoothed[frame_num, track_id]
                end_pos = smoothed[last_frame, track_id]
                if start_pos is None or end_pos is None:
                    continue

                distance_covered = measure_distance(start_pos, end_pos)
                time_elapsed = (last_frame - frame_num) / self.fps
                if time_elapsed <= 0:
                    continue

                speed_kmph = (distance_covered / time_elapsed) * 3.6
                if speed_kmph > self.max_speed_kmh:
                    self.rejected_windows += 1
                    for fn in range(frame_num, last_frame + 1):
                        tracks["players"][fn][track_id]["motion_rejected"] = True
                        tracks["players"][fn][track_id].pop("speed", None)
                    continue
                previous_distance = total_distance.get(track_id, 0.0)
                total_distance[track_id] = total_distance.get(track_id, 0.0) + distance_covered

                # +1 so the window's own last frame gets written too -- otherwise the
                # very last frame of the whole clip (where last_frame == n_frames - 1
                # and there's no further window to start from it) would never get a
                # speed/distance value at all. Consecutive windows' ranges touch at
                # exactly one frame (this window's last_frame == the next window's
                # frame_num); that frame simply gets overwritten by the next window's
                # value a moment later, which is harmless.
                for fn in range(frame_num, last_frame + 1):
                    if track_id in tracks["players"][fn]:
                        tracks["players"][fn][track_id].pop("motion_rejected", None)
                        tracks["players"][fn][track_id]["speed"] = speed_kmph
                        tracks["players"][fn][track_id]["distance"] = previous_distance + distance_covered * (fn-frame_num)/(last_frame-frame_num)

    def draw_frame_speed_and_distance(self, frame, frame_num, tracks):
        """
        Single-frame version of draw_speed_and_distance -- draws directly onto `frame`
        and returns it, for use by the streaming renderer (see
        utils/video_utils.py's render_video_streaming).
        """
        for _, track in tracks["players"][frame_num].items():
            if "speed" not in track:
                continue
            bbox = track["bbox"]  # Render in current-frame pixels, not stabilized coordinates.
            x, y = get_foot_position(bbox)
            y += 40
            cv2.putText(frame, f"{track['speed']:.1f} km/h", (int(x), int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
            y += 15
            cv2.putText(frame, f"{track['distance']:.1f} m", (int(x), int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
        return frame

    def draw_speed_and_distance(self, frames, tracks):
        """
        List-based version -- builds and returns a full second copy of the clip. For
        anything more than a couple hundred frames, prefer render_video_streaming()
        with draw_frame_speed_and_distance instead (see utils/video_utils.py). Kept
        here for backward compatibility.
        """
        output = []
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            frame = self.draw_frame_speed_and_distance(frame, frame_num, tracks)
            output.append(frame)
        return output
