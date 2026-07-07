"""Cursor detection from frames: template matching -> CursorTrack.

Uses a synthetic frame with a template pasted at a known location, so the match
is verifiable without any real recording. numpy is required (skips otherwise);
OpenCV is used when present, else the numpy fallback.
"""

from __future__ import annotations

import unittest

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

from demo2skill.video.statediff.cursor_detect import (
    TemplateCursorDetector,
    build_cursor_detector,
    detect_cursor_track,
    match_template,
)


@unittest.skipUnless(_HAS_NUMPY, "numpy not installed")
class MatchTemplateTest(unittest.TestCase):
    def _template(self):
        t = np.zeros((10, 8), dtype="float32")
        t[1:8, 1:4] = 255.0            # a distinctive bright blob
        return t

    def test_finds_template_at_known_location(self):
        img = np.full((120, 160), 30.0, dtype="float32")
        tmpl = self._template()
        px, py = 90, 40
        img[py:py + tmpl.shape[0], px:px + tmpl.shape[1]] = tmpl
        x, y, score = match_template(img, tmpl)
        self.assertEqual((x, y), (px, py))
        self.assertGreater(score, 0.9)


@unittest.skipUnless(_HAS_NUMPY, "numpy not installed")
class DetectorTest(unittest.TestCase):
    def test_locate_returns_hit_with_hotspot(self):
        from PIL import Image
        tmpl = np.zeros((10, 8), dtype="float32")
        tmpl[1:8, 1:4] = 255.0
        frame = np.full((120, 160), 30.0, dtype="float32")
        frame[40:50, 90:98] = tmpl
        img = Image.fromarray(frame.astype("uint8"), mode="L").convert("RGB")

        det = TemplateCursorDetector(tmpl, threshold=0.5, hotspot=(2, 3))
        hit = det.locate(img)
        self.assertIsNotNone(hit)
        self.assertEqual((hit.x, hit.y), (90 + 2, 40 + 3))

    def test_below_threshold_returns_none(self):
        from PIL import Image
        tmpl = np.zeros((10, 8), dtype="float32")
        tmpl[1:8, 1:4] = 255.0
        img = Image.new("RGB", (160, 120), (30, 30, 30))  # blank — no match
        det = TemplateCursorDetector(tmpl, threshold=0.9)
        self.assertIsNone(det.locate(img))


class FakeDetector:
    """Returns a fixed hit for every frame (moving right each call)."""

    def __init__(self):
        self.n = 0

    def locate(self, image):
        from demo2skill.video.statediff.cursor_detect import CursorHit
        self.n += 1
        return CursorHit(10 * self.n, 20, 0.99)


class TrackAssemblyTest(unittest.TestCase):
    def test_detect_cursor_track_builds_samples_per_frame(self):
        from demo2skill.video.video2action.frames import Frame, Frames
        # Frames need bytes; give each a 1x1 PNG so .bytes() is non-None.
        import io
        from PIL import Image
        png = io.BytesIO()
        Image.new("RGB", (4, 4), "white").save(png, format="PNG")
        data = png.getvalue()

        class _F(Frame):
            def bytes(self_inner):
                return data

        frames = Frames([_F(index=0, ms=0), _F(index=1, ms=1000)])
        track = detect_cursor_track(FakeDetector(), frames)
        self.assertEqual(len(track), 2)
        self.assertEqual((track.samples[0].x, track.samples[0].y), (10, 20))
        self.assertEqual(track.samples[1].x, 20)


class BuildSelectionTest(unittest.TestCase):
    def test_none_returns_none(self):
        self.assertIsNone(build_cursor_detector("none"))

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            build_cursor_detector("magic")


if __name__ == "__main__":
    unittest.main()
