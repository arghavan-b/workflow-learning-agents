"""Cursor detection from frames: the missing pixels -> cursor-position front.

`cursor.py` reasons over a stream of `{ms, x, y, clicking}` samples but does not
find the pointer in the image. This module produces those samples: it locates the
cursor in each frame by template matching, so an ordinary screen recording (no
instrumentation) can feed the same cursor-evidence the recorder provided.

    frames --[ CursorDetector ]--> CursorTrack --> state-diff IDM (clicks, coords)

`TemplateCursorDetector` matches a cursor template (a small crop of the pointer)
against each frame with normalized cross-correlation (OpenCV when available, a
NumPy fallback otherwise) and returns the tip location. Provide a real crop of
your OS cursor via `template_from_image` for best results; `synthetic_arrow`
is a rough fallback. Click detection is not attempted in v0 (`clicking=False`);
the IDM's `click_signature` still fires on cursor dwell, so clicks are inferred
from "the pointer arrived and settled."

Deps (numpy, optionally opencv) are imported lazily, so importing this module is
safe without them; install with the `cursor` extra.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Protocol, Tuple, runtime_checkable

from demo2skill.video.statediff.cursor import CursorSample, CursorTrack

logger = logging.getLogger("demo2skill.parser")


@dataclass
class CursorHit:
    x: int
    y: int
    confidence: float


@runtime_checkable
class CursorDetector(Protocol):
    def locate(self, image: Any) -> Optional[CursorHit]:
        """Return the cursor tip location in one frame, or None if not found."""
        ...


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised only with extra
        raise SystemExit(
            "The 'numpy' package is required for cursor detection.\n"
            "Install it with:  uv sync --extra cursor"
        ) from exc
    return np


def match_template(image_gray, template_gray) -> Tuple[int, int, float]:
    """Best (x, y, score) of ``template`` in ``image`` via normalized correlation.

    Uses OpenCV's ``matchTemplate`` when installed (fast); otherwise a NumPy
    sliding-window fallback. Both return the top-left of the best match.
    """

    np = _require_numpy()
    image_gray = np.asarray(image_gray, dtype="float32")
    template_gray = np.asarray(template_gray, dtype="float32")
    try:
        import cv2
        res = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _min, maxv, _minloc, maxloc = cv2.minMaxLoc(res)
        return int(maxloc[0]), int(maxloc[1]), float(maxv)
    except ImportError:
        # The NumPy fallback is O(W*H*w*h) — a full-res frame would hang. Downscale
        # the search to a bounded size, match, then map the location back.
        from PIL import Image
        max_edge = 640
        scale = min(1.0, max_edge / max(image_gray.shape))
        if scale < 1.0:
            ih, iw = image_gray.shape
            th, tw = template_gray.shape
            img_s = np.asarray(Image.fromarray(image_gray.astype("uint8")).resize(
                (max(1, int(iw * scale)), max(1, int(ih * scale)))), dtype="float32")
            tpl_s = np.asarray(Image.fromarray(template_gray.astype("uint8")).resize(
                (max(1, int(tw * scale)), max(1, int(th * scale)))), dtype="float32")
            x, y, score = _match_numpy(img_s, tpl_s, np)
            return int(x / scale), int(y / scale), score
        return _match_numpy(image_gray, template_gray, np)


def _match_numpy(image, template, np) -> Tuple[int, int, float]:
    ih, iw = image.shape
    th, tw = template.shape
    t = template - template.mean()
    tnorm = float(np.sqrt((t * t).sum())) + 1e-6
    best = (0, 0, -1.0)
    for y in range(ih - th + 1):
        for x in range(iw - tw + 1):
            patch = image[y:y + th, x:x + tw]
            p = patch - patch.mean()
            denom = float(np.sqrt((p * p).sum())) + 1e-6
            score = float((p * t).sum()) / (denom * tnorm)
            if score > best[2]:
                best = (x, y, score)
    return best


class TemplateCursorDetector:
    """Locate the cursor by matching a template crop against each frame."""

    def __init__(self, template, *, threshold: float = 0.5,
                 hotspot: Tuple[int, int] = (0, 0)) -> None:
        np = _require_numpy()
        # Normalize the template to a 2D grayscale float array.
        arr = np.asarray(template)
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
        self.template = arr.astype("float32")
        self.threshold = threshold
        self.hotspot = hotspot

    def locate(self, image: Any) -> Optional[CursorHit]:
        np = _require_numpy()
        gray = np.asarray(image.convert("L"), dtype="float32")
        if gray.shape[0] < self.template.shape[0] or gray.shape[1] < self.template.shape[1]:
            return None
        x, y, score = match_template(gray, self.template)
        if score < self.threshold:
            return None
        return CursorHit(x + self.hotspot[0], y + self.hotspot[1], score)


def detect_cursor_track(detector: CursorDetector, frames: Any) -> CursorTrack:
    """Run ``detector`` over a frame stream and assemble a `CursorTrack`.

    Frames where the cursor isn't found carry the last known position with
    ``visible=False``, so time windows are never empty.
    """

    from PIL import Image

    samples: List[CursorSample] = []
    last: Optional[Tuple[int, int]] = None
    total = len(frames.frames)
    found = 0
    for f in frames.frames:
        b = f.bytes() if hasattr(f, "bytes") else None
        if b is None:
            continue
        pil = Image.open(io.BytesIO(b)).convert("RGB")
        hit = detector.locate(pil)
        if hit is not None:
            last = (int(hit.x), int(hit.y))
            samples.append(CursorSample(ms=f.ms, x=last[0], y=last[1], visible=True))
            found += 1
        elif last is not None:
            samples.append(CursorSample(ms=f.ms, x=last[0], y=last[1], visible=False))
    logger.info("cursor detected in %d/%d frames", found, total)
    return CursorTrack(samples)


def template_from_image(path_or_image, *, hotspot: Tuple[int, int] = (2, 2)):
    """Build a (template, hotspot) from a cropped cursor image (a real crop is best)."""

    np = _require_numpy()
    from PIL import Image
    img = path_or_image if hasattr(path_or_image, "convert") else Image.open(path_or_image)
    return np.asarray(img.convert("L"), dtype="float32"), hotspot


def synthetic_arrow():
    """A rough arrow-pointer template (fallback when no real cursor crop is given)."""

    np = _require_numpy()
    from PIL import Image, ImageDraw
    img = Image.new("L", (16, 24), 0)
    d = ImageDraw.Draw(img)
    d.polygon([(1, 1), (1, 18), (5, 14), (8, 22), (11, 21), (8, 13), (14, 13)],
              fill=255)
    return np.asarray(img, dtype="float32"), (1, 1)


def build_cursor_detector(name: Optional[str], *, template_path: Optional[str] = None,
                          threshold: float = 0.5) -> Optional[CursorDetector]:
    """Construct a detector by name (``none`` -> ``None``)."""

    if not name or name == "none":
        return None
    if name == "template":
        if template_path:
            tmpl, hot = template_from_image(template_path)
        else:
            tmpl, hot = synthetic_arrow()
        return TemplateCursorDetector(tmpl, threshold=threshold, hotspot=hot)
    raise ValueError(f"unknown cursor detector {name!r} (use 'template' or 'none')")
