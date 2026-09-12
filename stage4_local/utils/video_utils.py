import cv2
import math


class VideoFrames:
    """Seekable video sequence that decodes frames on demand instead of retaining them.

    Sequential reads reuse the decoder; a new pass seeks back to its first frame.
    Slices allocate only their requested batch. Call close() when finished.
    """

    def __init__(self, video_path, target_width=1920):
        self.cap = cv2.VideoCapture(str(video_path))
        self.target_width = target_width
        self._next_frame = 0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        if not self.cap.isOpened() or self.frame_count <= 0:
            self.close()
            raise ValueError(f"Cannot read a nonempty video: {video_path}")
        if not math.isfinite(self.fps) or self.fps <= 0:
            self.close()
            raise ValueError(f"Video has invalid frame rate: {video_path}")

    def __len__(self):
        return self.frame_count

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self))) ]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index != self._next_frame:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError(f"Could not decode frame {index}")
        self._next_frame = index + 1
        h, w = frame.shape[:2]
        if self.target_width is not None and w > self.target_width:
            frame = cv2.resize(frame, (self.target_width, int(h * self.target_width / w)),
                               interpolation=cv2.INTER_AREA)
        return frame

    def close(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()


def read_video(video_path, target_width=1920):
    # Resize down while reading, not after — our clip is native 4K (3840x2160), and
    # holding all ~360 frames in memory at full 4K is ~9GB on its own, enough to crash
    # Colab's free-tier RAM by itself. target_width=1920 matches the resolution our
    # training data was stored at, so detection stays consistent with training too.
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        if target_width is not None and w > target_width:
            scale = target_width / w
            frame = cv2.resize(frame, (target_width, int(h * scale)), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    return frames


def get_native_frame(video_path, frame_idx):
    """
    Reads exactly ONE frame directly from the source video file at its original,
    un-downsampled resolution (e.g. native 4K even though read_video() gives back
    1920-wide frames for detection/tracking/drawing).

    Used only for team-color sampling: a player crop that's already tiny gets made
    even blurrier by the resize read_video() does to keep RAM usage sane, which
    contaminates shirt-color sampling with blended-in grass pixels. Re-reading just
    the handful of frames we actually need color from, at full detail, avoids that
    without holding the whole clip in memory at 4K.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
    return frame


def save_video(output_video_frames, output_video_path, fps=25):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w = output_video_frames[0].shape[:2]
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
    for frame in output_video_frames:
        out.write(frame)
    out.release()


def render_video_streaming(video_frames, output_path, draw_fns, fps=25):
    """
    Renders an annotated video WITHOUT ever holding a second full copy of the clip in
    memory: draws each frame, writes it straight to the video file, then moves on --
    one frame at a time.

    Prefer this over the pattern `save_video(some_drawer.draw_x(some_drawer2.draw_y(video_frames, ...), ...))`
    for any clip longer than a couple hundred frames. Each list-based draw_* method
    (Tracker.draw_annotations, SpeedAndDistanceEstimator.draw_speed_and_distance, ...)
    builds and returns a FULL new copy of the clip; chaining N of them, plus the
    original video_frames still being referenced, means roughly (N+1) full copies of
    every frame in the clip alive in RAM at the same time. That's exactly what
    crashes a Colab session with "used all available RAM" on a longer clip -- even
    when detection/tracking (the actual GPU-heavy step) already completed fine.

    video_frames: the original frames -- mutated in place, frame by frame, as they're
    drawn on and written out. Don't rely on them being unmodified afterward.
    draw_fns: list of callables, each `(frame, frame_num) -> frame`, applied in order
    to every frame before it's written -- e.g.
    [lambda f, n: tracker.draw_frame_annotations(f, n, tracks, team_ball_control),
     lambda f, n: speed_estimator.draw_frame_speed_and_distance(f, n, tracks)]
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w = video_frames[0].shape[:2]
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for frame_num, frame in enumerate(video_frames):
        for draw_fn in draw_fns:
            frame = draw_fn(frame, frame_num)
        out.write(frame)
    out.release()
