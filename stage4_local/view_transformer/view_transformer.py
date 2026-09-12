import sys
sys.path.append('/content')

import cv2
import numpy as np

from utils.bbox_utils import get_center_of_bbox, get_foot_position


class ViewTransformer:
    """
    Maps pixel positions from the video into real-world pitch coordinates (meters),
    via a homography computed once from a handful of known reference points -- e.g.
    the pitch's four corners -- whose pixel location you read off a frame, paired
    with their known real-world location on an actual pitch.

    This project's footage is a single, effectively static full-pitch overhead shot
    (the whole pitch is visible in every frame), so ONE homography, computed once,
    covers the entire clip -- unlike a panning/zooming broadcast camera, which would
    need this recomputed per frame or per shot.

    IMPORTANT: pixel_points must be given in order going around the pitch boundary
    (e.g. top-left, top-right, bottom-right, bottom-left) -- not paired diagonally --
    since they're also used to build the polygon that decides whether a given point
    is inside the calibrated pitch region at all.
    """

    def __init__(self, pixel_points, target_points):
        pixel_points = np.array(pixel_points, dtype=np.float32)
        target_points = np.array(target_points, dtype=np.float32)
        if len(pixel_points) < 4 or len(pixel_points) != len(target_points):
            raise ValueError("Need at least 4 matching pixel/target reference point pairs")

        self.pixel_polygon = pixel_points
        homography, _ = cv2.findHomography(pixel_points, target_points)
        if homography is None:
            raise ValueError("Could not compute a homography from the given reference points "
                              "-- check they aren't collinear or duplicated")
        self.perspective_transformer = homography

    def transform_point(self, point):
        """
        point: (x, y) pixel coordinate. Returns the corresponding (x, y) real-world
        pitch coordinate in meters, or None if the point falls outside the reference
        polygon. Extrapolating a homography far outside the region it was calibrated
        on gives meaningless (sometimes wildly wrong) results, so out-of-bounds points
        are refused rather than silently "transformed" into garbage.
        """
        p = (float(point[0]), float(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_polygon, p, False) >= 0
        if not is_inside:
            return None

        reshaped = np.array([point], dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(reshaped, self.perspective_transformer)
        return transformed.reshape(-1, 2)[0]

    def transform_tracks(self, tracks, pixel_scale=1.0):
        """
        Adds a "position_transformed" key (real-world [x, y] in meters, or None if
        outside the calibrated pitch region) to every player/ball track, using each
        object's camera-motion-adjusted position if camera_movement_estimator has
        already run (falls back to the raw bbox otherwise): foot position for players
        (where they're actually standing on the pitch), center for the ball.

        pixel_scale: multiply the extracted (bbox-derived) position by this factor
        before transforming, WITHOUT touching the stored bbox/adjusted_bbox itself.

        This matters because the homography's reference points (pixel_points passed
        to __init__) are read off a NATIVE-resolution frame (e.g. via get_native_frame),
        but the bboxes stored in `tracks` come from the tracker running on
        read_video()'s DOWNSAMPLED frames (default target_width=1920). If those two
        pixel spaces don't match, every position silently transforms into the wrong
        real-world location -- typically clustering everyone near one corner of the
        pitch instead of spanning it (and, less visibly, scaling every speed/distance
        number by roughly the same downsample ratio). Pass
        pixel_scale = native_frame_width / downsampled_frame_width to correct for it;
        leave it at the default 1.0 when bboxes are already in the same pixel space
        the homography was calibrated in.
        """
        for obj_type in ["players", "ball"]:
            for frame_num, frame_tracks in enumerate(tracks[obj_type]):
                for track_id, track in frame_tracks.items():
                    bbox = track.get("adjusted_bbox", track["bbox"])
                    position = get_foot_position(bbox) if obj_type == "players" else get_center_of_bbox(bbox)
                    scaled_position = (position[0] * pixel_scale, position[1] * pixel_scale)
                    transformed = self.transform_point(scaled_position)
                    tracks[obj_type][frame_num][track_id]["position_transformed"] = (
                        transformed.tolist() if transformed is not None else None
                    )
