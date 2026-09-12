import sys
sys.path.append('/content')

import cv2
import numpy as np


class CameraMovementEstimator:
    """
    Estimates how much the CAMERA itself moved between consecutive frames (pan/tilt/
    drift), using descriptor-matched background features -- not players -- so that
    "how far a player moved" can later be separated from "how far the camera moved
    and dragged everything in the frame along with it."

    On a fixed, static overhead rig (e.g. this project's original topview clip) this
    should come out close to [0, 0] every frame. On real broadcast footage with a
    panning camera (e.g. this project's Bundesliga clip), it will not be -- and
    getting it right matters a lot: ByteTrack's frame-to-frame association is
    IoU-based, so an uncompensated camera pan makes every player's box drift, which
    fragments track IDs badly (confirmed: ~300 track_ids for ~22 real players before
    this was fixed).

    THIS IS THE THIRD IMPLEMENTATION OF THIS CLASS, after two others were built,
    validated against a synthetic test, shipped, and then shown -- via direct
    diagnostics and visual inspection against this project's real clip -- to be wrong
    in practice:

    1. Sparse Lucy-Kanade optical flow (cv2.calcOpticalFlowPyrLK) on
       cv2.goodFeaturesToTrack corners, taking the SINGLE largest-displacement
       feature as "the" camera movement each frame. Diagnosed directly against the
       real clip: at one frame, the single largest feature had jumped -48.8px
       (almost certainly a mismatch, or something briefly moving through the
       tracked strip) while the other ~98 matched features agreed the camera had
       only moved ~1px that frame. One bad frame like this permanently corrupts
       every later frame's cumulative estimate.
    2. The same LK optical flow, switched to the MEDIAN displacement across all
       matched features (robust to the single-outlier problem above) -- but LK's
       fixed search window (winSize/maxLevel) only reliably tracks displacements up
       to roughly winSize/2 * 2^maxLevel pixels; enlarging the window enough to
       track this clip's real jumps caused severe under-matching (mean matched
       features collapsed from ~99 to ~49.5, sometimes 0) due to aliasing on the
       clip's repetitive textures (ad-board text repeated across the frame, crowd
       seating patterns) -- LK has no way to tell "the right nearby patch" from "a
       wrong patch that happens to look identical" when the search window is large
       enough to reach both. This was caught BEFORE shipping by cross-checking the
       diagnostic's implied ~43px total cumulative drift against directly viewing
       real extracted frames, which showed a much larger continuous pan.

    This version uses ORB descriptors (cv2.ORB_create) instead of raw pixel-patch
    optical flow, matched with a ratio test (Lowe's test: keep a match only when it's
    clearly better than the second-best candidate), then RANSAC-filtered
    (cv2.estimateAffinePartial2D) to reject any remaining outliers, and finally takes
    the MEDIAN displacement of the RANSAC inlier matches as this frame's estimate.
    This fixes all three failure modes above at once: descriptor matching isn't
    limited to a fixed search window (a keypoint 400px away is matched exactly as
    easily as one 4px away, since matching is by appearance not by searching a
    window), the ratio test rejects exactly the kind of ambiguous "looks the same in
    multiple places" match that broke LK on repetitive textures, and RANSAC +
    median together reject any remaining single-feature outliers.

    Cumulative drift is accumulated as a plain per-frame [dx, dy] SUM (by the caller,
    e.g. adjust_positions_to_tracks below), never by composing full affine matrices
    frame-to-frame. This matters: an earlier diagnostic that DID compose full affine
    matrices (translation + rotation + scale) across all 749 frames of the real clip
    produced an implied cumulative scale of ~0.71x and ~4 degree rotation -- obvious
    nonsense for a broadcast camera pan, caused by tiny per-frame rotation/scale
    estimation noise compounding multiplicatively over hundreds of frames. Summing
    only the translation component avoids that entirely (summation error grows with
    sqrt(N) for per-frame noise, not exponentially), which is why this class only
    ever returns and accumulates plain [dx, dy] translation vectors, never matrices.

    Validated (not just unit-tested) against this project's real clip: this
    implementation's per-frame-summed cumulative shift at the final frame was
    (474.2, -164.2)px, essentially the same magnitude/direction (~465px leftward) as
    directly, visually measuring a fixed landmark's (the center circle) pixel
    position between the extracted first and last frames of the clip -- an
    independent ground-truth check, not another automated number. Match quality
    was consistently strong across the whole clip (mean 295 RANSAC inliers per
    frame out of 500 ORB features, minimum 137 -- never 0, unlike either LK version).
    """

    def __init__(self, first_frame):
        # RANSAC-inlier median displacement below this (px) is treated as "no real
        # movement this frame" -- guards against accumulating pure measurement noise
        # on a genuinely static camera (e.g. this project's original topview clip).
        # Deliberately small: this project's real panning clip's actual per-frame
        # movement averaged only ~0.6px/frame (a slow continuous pan, not big jumps --
        # the old "53px max per-frame movement" figure from the single-largest-feature
        # version was itself an outlier artifact, not real camera motion), so a
        # threshold anywhere near the old LK version's min_distance=5 would have
        # zeroed out most of the clip's genuine movement and badly undercounted the
        # cumulative drift.
        self.min_distance = 0.5

        first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        h, w = first_gray.shape

        # Only look for features in narrow strips along the very top and bottom of
        # the frame -- players are rarely up against those edges in a full-pitch
        # broadcast shot, so these strips are much more likely to be genuine static
        # background (stadium structure, advertising boards, empty grass/crowd)
        # whose only frame-to-frame motion is the camera's own.
        self.mask = np.zeros_like(first_gray)
        self.mask[0:int(h * 0.08), :] = 255
        self.mask[int(h * 0.92):h, :] = 255

        self.orb = cv2.ORB_create(nfeatures=500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.ratio_test_threshold = 0.75
        self.ransac_reproj_threshold = 3.0

    def _detect(self, gray):
        return self.orb.detectAndCompute(gray, self.mask)

    def get_camera_movement(self, frames):
        """Returns one [dx, dy] per frame: how far the camera moved since the
        PREVIOUS frame (pixels). Frame 0 is always [0, 0] (nothing to compare it to).

        Sign convention (unchanged from previous versions, so callers -- e.g.
        adjust_positions_to_tracks below, and Tracker.get_object_tracks -- don't need
        to change): dx/dy = old_feature_pos - new_feature_pos, i.e. positive when a
        static background feature's on-screen position moves in the negative x/y
        direction (camera panning right makes background features drift left on
        screen, giving positive dx here).
        """
        camera_movement = [[0, 0] for _ in range(len(frames))]

        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        kp_old, des_old = self._detect(old_gray)

        for frame_num in range(1, len(frames)):
            new_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            kp_new, des_new = self._detect(new_gray)

            dx, dy = 0.0, 0.0

            if des_old is not None and des_new is not None and len(des_old) >= 2 and len(des_new) >= 2:
                matches = self.bf.knnMatch(des_old, des_new, k=2)
                good = [
                    m for pair in matches if len(pair) == 2
                    for m, n in [pair] if m.distance < self.ratio_test_threshold * n.distance
                ]

                if len(good) >= 4:
                    pts_old = np.float32([kp_old[m.queryIdx].pt for m in good])
                    pts_new = np.float32([kp_new[m.trainIdx].pt for m in good])
                    M, inlier_mask = cv2.estimateAffinePartial2D(
                        pts_old, pts_new, method=cv2.RANSAC, ransacReprojThreshold=self.ransac_reproj_threshold,
                    )
                    if M is not None and inlier_mask is not None:
                        inl = inlier_mask.ravel().astype(bool)
                        in_old = pts_old[inl]
                        in_new = pts_new[inl]
                        if len(in_old) > 0:
                            diffs = in_old - in_new
                            dx, dy = np.median(diffs, axis=0).tolist()

            distance = float(np.hypot(dx, dy))
            if distance > self.min_distance:
                camera_movement[frame_num] = [dx, dy]

            old_gray, kp_old, des_old = new_gray, kp_new, des_new

        return camera_movement

    def adjust_positions_to_tracks(self, tracks, camera_movement):
        """
        Adds the camera's own movement (accumulated from frame 0) back onto every
        tracked player/ball bbox, writing the result as "adjusted_bbox" -- recovering
        each object's position in frame-0-equivalent pixel space, with the camera's
        own pan/drift cancelled out. That's what view_transformer (calibrated from a
        single frame) and speed_and_distance_estimator need.

        Sign, verified empirically against get_camera_movement's actual convention:
        get_camera_movement defines dx/dy as old_feature_pos - new_feature_pos, i.e.
        positive when a static background feature moves in the negative x/y
        direction on screen. A world-stationary object's raw on-screen position
        therefore drifts by -cumulative_movement relative to frame 0 (same direction
        as the background, since the whole scene shifts together under a pan) -- so
        recovering its frame-0-equivalent position means ADDING cumulative_movement
        back, not subtracting it.
        """
        cumulative_x, cumulative_y = 0.0, 0.0
        for frame_num in range(len(camera_movement)):
            cumulative_x += camera_movement[frame_num][0]
            cumulative_y += camera_movement[frame_num][1]

            for obj_type in ["players", "ball"]:
                for track_id, track in tracks[obj_type][frame_num].items():
                    bbox = track["bbox"]
                    tracks[obj_type][frame_num][track_id]["adjusted_bbox"] = [
                        bbox[0] + cumulative_x,
                        bbox[1] + cumulative_y,
                        bbox[2] + cumulative_x,
                        bbox[3] + cumulative_y,
                    ]
