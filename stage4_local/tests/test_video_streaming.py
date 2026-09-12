import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.video_utils import VideoFrames
from trackers.tracker import Tracker


class StreamingTests(unittest.TestCase):
    def test_seeking_batches_and_repeated_passes_preserve_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "test.avi")
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
            self.assertTrue(writer.isOpened())
            for i in range(12):
                writer.write(np.full((48, 64, 3), i * 20, dtype=np.uint8))
            writer.release()
            with VideoFrames(path, target_width=32) as frames:
                self.assertEqual(len(frames), 12)
                self.assertEqual(frames.fps, 30)
                self.assertEqual(frames[0].shape, (24, 32, 3))
                self.assertAlmostEqual(float(frames[-1].mean()), 220, delta=3)
                self.assertEqual(len(frames[3:6]), 3)
                first = [float(frame.mean()) for frame in frames]
                second = [float(frame.mean()) for frame in frames]
                self.assertEqual(first, second)
                self.assertLess(first[0], first[-1])

    def test_cached_drawing_does_not_import_inference_runtime(self):
        tracker = Tracker(None)
        self.assertIsNone(tracker.model)
        self.assertNotIn("torch", sys.modules)
        with self.assertRaisesRegex(ValueError, "No cached tracks"):
            tracker.get_object_tracks([])

    def test_missing_video_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "Cannot read"):
            VideoFrames("nonexistent-pitch-vision-test.mp4")


if __name__ == "__main__":
    unittest.main()
