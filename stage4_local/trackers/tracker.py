import os
import pickle
import sys

import cv2
import numpy as np

sys.path.append('/content')
from utils.bbox_utils import get_center_of_bbox, get_bbox_width


class Tracker:
    def __init__(self, model_path, fps=25, occlusion_buffer_seconds=3.0):
        # Cached downstream analysis and drawing do not need the inference runtime.
        self.model = None
        self.tracker = None
        if model_path is None:
            return
        import supervision as sv
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        # Defaults (lost_track_buffer=30, frame_rate=30) give ByteTrack only ~1s of
        # patience before it gives up on a lost track and assigns a new ID on
        # reappearance -- too short for a player who gets blocked by another player
        # for a second or two, which is routine in broadcast footage (confirmed: this
        # is exactly what was reported happening -- a track's number changing after
        # one player passed in front of another). frame_rate is also set explicitly
        # rather than left at the default-30 assumption, since lost_track_buffer is
        # converted to an actual frame count using it. occlusion_buffer_seconds=3.0 is
        # a deliberate, generous tolerance for brief occlusions -- the tradeoff is a
        # small risk of stitching two DIFFERENT players together if one leaves and
        # another enters roughly the same spot within that window, which is rarer
        # than a simple pass-in-front occlusion.
        self.tracker = sv.ByteTrack(
            # supervision scales this 30-fps reference buffer by frame_rate / 30.
            lost_track_buffer=int(occlusion_buffer_seconds * 30),
            frame_rate=fps,
        )

    def detect_frames(self, frames, conf=0.2, imgsz=1280, batch_size=20):
        # imgsz=1280 matches training — inference at the default 640 would shrink the
        # ball back down below what the model was actually trained to recognize.
        for i in range(0, len(frames), batch_size):
            batch = self.model.predict(frames[i:i + batch_size], conf=conf, imgsz=imgsz, verbose=False)
            yield from batch

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None, camera_movement=None):
        """
        camera_movement: optional list of [dx, dy] per frame (CameraMovementEstimator's
        output -- each frame's shift since the PREVIOUS frame, frame 0 = [0, 0]).

        On a panning camera, every detection's apparent pixel position shifts by the
        camera's own motion each frame, on top of whatever the player actually did.
        ByteTrack's frame-to-frame association is IoU-based, so a big camera jump can
        push a stationary player's box far enough from its predicted position that the
        match fails and a brand-new track_id gets assigned -- confirmed on this
        project's broadcast clip (a 53px max per-frame camera shift produced ~180
        distinct track_ids for ~22 real players) and reproduced in a synthetic test
        (24 IDs for 3 real players uncompensated, vs 3 IDs compensated, same
        panning pattern). Passing camera_movement shifts a COPY of each frame's
        detections into frame-0-relative ("camera-stabilized") pixel space before
        handing them to ByteTrack, so association only has to account for real player
        motion. The bbox actually STORED is shifted back to real per-frame pixel space
        afterward, so drawing/downstream code is unaffected -- this only changes what
        ByteTrack sees internally. When camera_movement is None (the default), nothing
        changes from the original behavior.
        """
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        if self.model is None:
            raise ValueError("No cached tracks found; provide a model for fresh inference")
        import supervision as sv

        detections = self.detect_frames(frames)

        tracks = {"players": [], "ball": []}

        cumulative_x, cumulative_y = 0.0, 0.0
        for frame_num, detection in enumerate(detections):
            if camera_movement is not None:
                cumulative_x += camera_movement[frame_num][0]
                cumulative_y += camera_movement[frame_num][1]

            cls_names = detection.names  # {0: 'player', 1: 'ball'}
            cls_names_inv = {v: k for k, v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)
            # Keep keepers in the observed-player set. They are not identified or
            # rated as specialist goalkeepers by this activity model.
            goalkeeper_id = cls_names_inv.get('goalkeeper')
            player_id = cls_names_inv.get('player')
            if goalkeeper_id is not None and player_id is not None:
                detection_supervision.class_id[detection_supervision.class_id == goalkeeper_id] = player_id

            if camera_movement is not None and len(detection_supervision) > 0:
                # + (not -): see CameraMovementEstimator.adjust_positions_to_tracks'
                # docstring for the empirically-verified sign -- a world-stationary
                # object's raw on-screen position drifts by -cumulative relative to
                # frame 0, so adding cumulative back recovers its stable position.
                stabilized_xyxy = detection_supervision.xyxy.copy()
                stabilized_xyxy[:, [0, 2]] += cumulative_x
                stabilized_xyxy[:, [1, 3]] += cumulative_y
                detection_for_tracking = sv.Detections(
                    xyxy=stabilized_xyxy,
                    confidence=detection_supervision.confidence,
                    class_id=detection_supervision.class_id,
                )
            else:
                detection_for_tracking = detection_supervision

            detection_with_tracks = self.tracker.update_with_detections(detection_for_tracking)

            tracks["players"].append({})
            tracks["ball"].append({})

            # players get persistent track_ids from ByteTrack
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                if camera_movement is not None:
                    # undo the stabilization shift (inverse of the += above) -- store
                    # the real per-frame pixel bbox
                    bbox = [bbox[0] - cumulative_x, bbox[1] - cumulative_y, bbox[2] - cumulative_x, bbox[3] - cumulative_y]
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                if cls_id == cls_names_inv.get('player'):
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

            # the ball gets a hardcoded track_id of 1 — there's only ever one, no need to
            # track its identity across frames, just its position. Uses the ORIGINAL
            # (un-stabilized) detections since it never goes through the tracker's
            # association at all -- the camera_movement shift above only ever affected
            # what ByteTrack saw internally.
            best_ball_confidence = -1.0
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                confidence = float(frame_detection[2])
                if cls_id == cls_names_inv.get('ball') and confidence > best_ball_confidence:
                    best_ball_confidence = confidence
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4,
        )

        rect_w, rect_h = 40, 20
        x1_rect = x_center - rect_w // 2
        x2_rect = x_center + rect_w // 2
        y1_rect = (y2 - rect_h // 2) + 15
        y2_rect = (y2 + rect_h // 2) + 15

        if track_id is not None:
            cv2.rectangle(frame, (int(x1_rect), int(y1_rect)), (int(x2_rect), int(y2_rect)), color, cv2.FILLED)
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10
            cv2.putText(frame, f"{track_id}", (int(x1_text), int(y1_rect + 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)
        points = np.array([[x, y], [x - 10, y - 20], [x + 10, y - 20]])
        cv2.drawContours(frame, [points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [points], 0, (0, 0, 0), 2)
        return frame

    def draw_frame_annotations(self, frame, frame_num, tracks, team_ball_control):
        """
        Single-frame version of draw_annotations -- draws directly onto `frame` (the
        caller owns whether that's safe to mutate) and returns it. This is what the
        memory-safe streaming renderer (utils/video_utils.py's render_video_streaming)
        calls per frame, instead of ever building a second full copy of the clip.
        """
        player_dict = tracks["players"][frame_num]
        ball_dict = tracks["ball"][frame_num]

        for track_id, player in player_dict.items():
            color = player.get("team_color", (0, 0, 255))
            frame = self.draw_ellipse(frame, player["bbox"], color, track_id)
            if player.get("has_ball", False):
                frame = self.draw_triangle(frame, player["bbox"], (0, 0, 255))

        for _, ball in ball_dict.items():
            frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

        if len(team_ball_control) > 0 and frame_num < len(team_ball_control):
            so_far = team_ball_control[:frame_num + 1]
            t1 = int((so_far == 1).sum())
            t2 = int((so_far == 2).sum())
            total = t1 + t2
            if total > 0:
                cv2.putText(frame, f"Team 1 Ball Control: {t1 / total * 100:.1f}%", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                cv2.putText(frame, f"Team 2 Ball Control: {t2 / total * 100:.1f}%", (50, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        """
        List-based version -- builds and returns a full second copy of the clip.
        Fine for a short clip; for anything more than a couple hundred frames, prefer
        render_video_streaming() in utils/video_utils.py with draw_frame_annotations,
        which writes each frame straight to disk instead of holding it. Kept here for
        backward compatibility (and it's what draw_frame_annotations now factors out of).
        """
        output_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            frame = self.draw_frame_annotations(frame, frame_num, tracks, team_ball_control)
            output_frames.append(frame)
        return output_frames
