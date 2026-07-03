"""Frame extraction: sampling strategies, real timestamps, format tolerance.

The ffmpeg/ffprobe calls themselves need a binary, so the pure parsing helpers
(timestamp + probe parsing, filter construction) are tested directly, and the
directory loader is tested end to end. ``from_video`` is exercised only for its
argument validation and its clean error when ffmpeg is absent.
"""

from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from demo2skill.video.video2action.frames import (
    Frames,
    _filter_for,
    _parse_probe,
    _parse_pts_ms,
)


class TimestampParsingTest(unittest.TestCase):
    def test_pts_times_become_ms_in_order(self):
        text = (
            "frame:0    pts:0        pts_time:0\n"
            "lavfi.scene_score=0.00\n"
            "frame:1    pts:1500     pts_time:1.5\n"
            "frame:2    pts:3200     pts_time:3.204\n"
        )
        self.assertEqual(_parse_pts_ms(text), [0, 1500, 3204])

    def test_empty_output_is_empty(self):
        self.assertEqual(_parse_pts_ms(""), [])


class ProbeParsingTest(unittest.TestCase):
    def test_parses_resolution_fps_duration(self):
        out = "width=1280\nheight=720\nr_frame_rate=30/1\nduration=42.123\n"
        w, h, fps, dur = _parse_probe(out)
        self.assertEqual((w, h), (1280, 720))
        self.assertAlmostEqual(fps, 30.0)
        self.assertAlmostEqual(dur, 42.123)

    def test_tolerates_missing_and_fractional_rate(self):
        out = "width=N/A\nheight=1080\nr_frame_rate=30000/1001\n"
        w, h, fps, dur = _parse_probe(out)
        self.assertIsNone(w)
        self.assertEqual(h, 1080)
        self.assertAlmostEqual(fps, 29.97, places=2)
        self.assertIsNone(dur)


class FilterConstructionTest(unittest.TestCase):
    def test_modes_map_to_expected_filters(self):
        tf = Path("/tmp/times.txt")
        self.assertIn("fps=1.5", _filter_for("fps", fps=1.5, scene_threshold=0.3, times_file=tf))
        self.assertIn("pict_type", _filter_for("keyframes", fps=1, scene_threshold=0.3, times_file=tf))
        scene = _filter_for("scene", fps=1, scene_threshold=0.4, times_file=tf)
        self.assertIn("scene", scene)
        self.assertIn("0.4", scene)
        self.assertTrue(all("metadata=print" in _filter_for(m, fps=1, scene_threshold=0.3, times_file=tf)
                            for m in ("fps", "keyframes", "scene")))


class FromDirTest(unittest.TestCase):
    def test_reads_mixed_image_formats_in_order(self):
        with TemporaryDirectory() as d:
            for name in ("frame_000002.jpg", "frame_000000.png", "frame_000001.jpeg"):
                (Path(d) / name).write_bytes(b"x")
            frames = Frames.from_dir(d, fps=2.0)
            self.assertEqual(len(frames), 3)
            self.assertEqual([f.ms for f in frames.frames], [0, 500, 1000])


class FromVideoGuardsTest(unittest.TestCase):
    def test_bad_sample_mode_rejected(self):
        with self.assertRaises(ValueError):
            Frames.from_video("x.mp4", sample="nope")

    def test_missing_ffmpeg_raises_clear_error(self):
        if shutil.which("ffmpeg") is not None:
            self.skipTest("ffmpeg present; absence path not exercised here")
        with self.assertRaises(RuntimeError) as ctx:
            Frames.from_video("x.mp4")
        self.assertIn("ffmpeg", str(ctx.exception))


class RealFfmpegIntegrationTest(unittest.TestCase):
    """Exercise the actual decode path when ffmpeg is available."""

    def setUp(self):
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg not installed")

    def test_uniform_sampling_extracts_frames_with_real_timestamps(self):
        import subprocess

        with TemporaryDirectory() as d:
            clip = Path(d) / "clip.mp4"
            subprocess.run(
                ["ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
                 "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=30", str(clip)],
                check=True, capture_output=True,
            )
            frames = Frames.from_video(clip, fps=2.0, sample="fps", out_dir=Path(d) / "out")
            self.assertGreaterEqual(len(frames), 3)          # ~2s at 2fps
            self.assertEqual((frames.width, frames.height), (320, 240))
            ms = [f.ms for f in frames.frames]
            self.assertEqual(ms, sorted(ms))                  # monotonic
            self.assertEqual(ms[0], 0)
            self.assertTrue(all(f.path and f.path.exists() for f in frames.frames))


if __name__ == "__main__":
    unittest.main()
