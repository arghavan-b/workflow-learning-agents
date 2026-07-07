"""OCR fill: read the values ScreenVLM leaves blank inside input fields.

ScreenVLM detects input boxes and static labels but was trained on freshly
rendered pages, so it does not transcribe *user-typed* field contents (see
`screenvlm.py`). The state-diff IDM needs those values to detect `type` actions.
This module closes that gap: given ScreenVLM's elements and the original
(full-resolution) frame, it OCRs each editable element's crop and fills its
`value`.

The OCR backend is pluggable via the `OCR` protocol; `TesseractOCR` is the
lightweight reference (needs the `ocr` extra plus the system `tesseract`
binary). `ocr_fill_values` is a pure function so the fill logic is testable
without a real OCR engine.
"""

from __future__ import annotations

import logging
import re
from typing import Any, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger("demo2skill.parser")

# Generic placeholders an empty field commonly shows; treated as "no value".
_DEFAULT_PLACEHOLDERS = {
    "title", "search", "search...", "type here", "type your description here",
    "add a title", "add a description", "add a comment",
}


def _norm(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


_CJK = re.compile(r"[　-〿぀-ヿ㐀-䶿一-鿿＀-￯]")


def _has_cjk(text: str) -> bool:
    return bool(_CJK.search(text or ""))


@runtime_checkable
class OCR(Protocol):
    """Read text from a cropped image (one input field)."""

    def read(self, image: Any) -> str:
        ...


class TesseractOCR:
    """`OCR` backed by Tesseract via pytesseract.

    Install with the ``ocr`` extra (``uv sync --extra ocr``) *and* the system
    binary (e.g. ``brew install tesseract``). PSM 6 assumes a uniform block of
    text, which suits single-line inputs and small textareas.
    """

    def __init__(self, lang: str = "eng", psm: int = 6) -> None:
        try:
            import pytesseract
        except ImportError as exc:  # pragma: no cover - exercised only with extra
            raise SystemExit(
                "The 'pytesseract' package is required for OCR value-fill.\n"
                "Install it with:  uv sync --extra ocr\n"
                "and the engine itself, e.g.:  brew install tesseract"
            ) from exc
        # Fail fast (before the model loads) if the engine binary is missing,
        # with an actionable message instead of a crash mid-run.
        try:
            pytesseract.get_tesseract_version()
        except Exception as exc:  # TesseractNotFoundError and friends
            raise SystemExit(
                "The 'tesseract' engine binary is not on your PATH.\n"
                "Install it, e.g.:  brew install tesseract   (macOS)\n"
                "                   sudo apt-get install tesseract-ocr   (Linux)"
            ) from exc
        self._pt = pytesseract
        self.lang = lang
        self.config = f"--psm {psm}"

    def read(self, image: Any) -> str:
        text = self._pt.image_to_string(image, lang=self.lang, config=self.config)
        return " ".join(text.split()).strip()


class EasyOCRBackend:
    """`OCR` backed by EasyOCR — fully pip-installable, no system binary.

    Install with ``uv pip install easyocr`` (reuses the torch already pulled in
    for ScreenVLM). The first run downloads its detection/recognition models
    (~64MB). This is the no-brew alternative to Tesseract.
    """

    def __init__(self, langs=("en",), gpu: bool = False) -> None:
        try:
            import easyocr
            import numpy as np
        except ImportError as exc:  # pragma: no cover - exercised only with extra
            raise SystemExit(
                "The 'easyocr' package is required for this OCR backend.\n"
                "Install it with:  uv pip install easyocr   (no system binary needed)"
            ) from exc
        self._np = np
        self._reader = easyocr.Reader(list(langs), gpu=gpu)

    def read(self, image: Any) -> str:
        arr = self._np.array(image.convert("RGB"))
        lines = self._reader.readtext(arr, detail=0, paragraph=True)
        return " ".join(str(t).strip() for t in lines).strip()


class PaddleOCRBackend:
    """`OCR` backed by PaddleOCR (PP-OCRv5/v6), pip-installable, no system binary.

    Install with ``uv pip install paddleocr paddlepaddle`` (CPU) — or the GPU
    ``paddlepaddle-gpu``. Extra keyword args are forwarded to ``PaddleOCR`` so you
    can pin the recognizer, e.g. ``PaddleOCRBackend(ocr_version="PP-OCRv5")``.
    Result parsing tolerates both the 2.x ``.ocr()`` list shape and the 3.x
    ``.predict()`` dict shape.
    """

    def __init__(self, lang: str = "en", drop_score: float = 0.6, **paddle_kwargs) -> None:
        try:
            import numpy as np
            from paddleocr import PaddleOCR
        except ImportError as exc:  # pragma: no cover - exercised only with extra
            raise SystemExit(
                "The 'paddleocr' package is required for this OCR backend.\n"
                "Install it with:  uv pip install paddleocr paddlepaddle\n"
                "(use 'paddlepaddle-gpu' for GPU)"
            ) from exc
        self._np = np
        self._ocr = PaddleOCR(lang=lang, **paddle_kwargs)
        self.drop_score = drop_score
        self._english = str(lang).startswith("en")

    def read(self, image: Any) -> str:
        arr = self._np.array(image.convert("RGB"))
        result, last_exc = None, None
        for call in ("predict", "ocr"):
            fn = getattr(self._ocr, call, None)
            if fn is None:
                continue
            try:
                result = fn(arr)
                break
            except Exception as exc:  # try the other entrypoint / signature
                last_exc = exc
        if result is None and last_exc is not None:   # a real failure, not empty text
            logger.warning("PaddleOCR read failed: %s", last_exc)
        return self._join(result, drop_score=self.drop_score, drop_cjk=self._english)

    @staticmethod
    def _rec(page: Any):
        """Extract (texts, scores) from a 3.x result page (dict / OCRResult /
        object with .json), or ``None`` if this page is a 2.x detection list."""

        if isinstance(page, dict):
            return page.get("rec_texts") or [], page.get("rec_scores") or []
        try:                                        # 3.x OCRResult is dict-like
            texts = page["rec_texts"]
            try:
                scores = page["rec_scores"]
            except Exception:
                scores = []
            return texts, scores
        except Exception:
            pass
        j = getattr(page, "json", None)
        if isinstance(j, dict):
            res = j.get("res", j)
            return res.get("rec_texts") or [], res.get("rec_scores") or []
        return None

    @staticmethod
    def _join(result: Any, drop_score: float = 0.0, drop_cjk: bool = False) -> str:
        def _keep(text: str, score) -> bool:
            if not text or not str(text).strip():
                return False
            if score is not None and score < drop_score:
                return False
            if drop_cjk and _has_cjk(text):   # CJK in an English session = misread
                return False
            return True

        lines: List[str] = []
        for page in result or []:
            if page is None:
                continue
            rec = PaddleOCRBackend._rec(page)               # 3.x (v5/v6) shapes
            if rec is not None:
                texts, scores = rec
                scores = list(scores) + [None] * (len(texts) - len(scores))
                lines += [str(t) for t, s in zip(texts, scores) if _keep(t, s)]
                continue
            for det in page:                                # 2.x: [box, (text, conf)]
                try:
                    info = det[1]
                    if isinstance(info, (list, tuple)):
                        text, score = info[0], (info[1] if len(info) > 1 else None)
                    else:
                        text, score = info, None
                    if _keep(text, score):
                        lines.append(str(text))
                except (IndexError, TypeError):
                    continue
        return " ".join(s.strip() for s in lines if s and s.strip()).strip()


def make_ocr(name: Optional[str], *, ocr_version: Optional[str] = None) -> Optional[OCR]:
    """Construct an OCR backend by name (``none`` -> ``None``).

    ``ocr_version`` (e.g. ``"PP-OCRv6"``) applies to the ``paddle`` backend only.
    """

    if not name or name == "none":
        return None
    if name == "tesseract":
        return TesseractOCR()
    if name == "easyocr":
        return EasyOCRBackend()
    if name == "paddle":
        return PaddleOCRBackend(ocr_version=ocr_version) if ocr_version else PaddleOCRBackend()
    raise ValueError(
        f"unknown OCR backend {name!r} (use 'tesseract', 'easyocr', 'paddle', or 'none')")


def ocr_fill_values(elements: List[Any], image: Any, ocr: OCR,
                    *, inset: int = 2, upscale: int = 3, min_size: int = 6,
                    reject_texts: Optional[set] = None) -> List[Any]:
    """Fill each editable element's ``value`` by OCR-ing its crop of ``image``.

    Precision measures, since ScreenVLM's field boxes can be coarse:

    * **inset** the crop inward (avoid borders and adjacent labels), and
      **upscale** it before OCR (small field text reads far better enlarged);
    * **reject label/placeholder echoes** — if the OCR result matches a nearby
      static text (a label/button) or a known placeholder, it isn't typed
      content, so leave the field empty. This is what stops a title field's value
      from coming back as its own label ("Add a title").

    ``image`` is the original full-resolution frame. Only editable elements
    without a value are touched.
    """

    placeholders = {_norm(t) for t in (reject_texts or _DEFAULT_PLACEHOLDERS)}
    placeholders.discard("")

    width, height = image.size
    for el in elements:
        if not getattr(el, "editable", False):
            continue
        if el.value and str(el.value).strip():
            continue
        x1, y1, x2, y2 = el.bbox
        w, h = x2 - x1, y2 - y1
        if w < min_size or h < min_size:
            continue
        ix, iy = min(inset, w // 4), min(inset, h // 4)
        box = (max(0, x1 + ix), max(0, y1 + iy), min(width, x2 - ix), min(height, y2 - iy))
        if box[2] - box[0] < 2 or box[3] - box[1] < 2:
            continue
        crop = image.crop(box)
        if upscale and upscale > 1:
            crop = crop.resize((crop.width * upscale, crop.height * upscale))
        try:
            text = ocr.read(crop)
        except Exception as exc:  # a bad crop must not kill the whole run
            logger.warning("OCR failed on %s: %s", getattr(el, "id", "?"), exc)
            continue
        if not text:
            continue
        # Bug 5: reject only echoes of text *near this field* (its label row) plus
        # generic placeholders — not any static text anywhere on the screen, or a
        # real typed value that happens to equal a distant button would be lost.
        reject = placeholders | _nearby_texts(el, elements)
        if _norm(text) in reject:
            logger.info("  OCR echo rejected on %s: %r (matches a nearby label)",
                        getattr(el, "id", "?"), text)
            continue
        el.value = text
        if not el.text:
            el.text = text

    _dedup_overlapping_values(elements)
    return elements


def _nearby_texts(field: Any, elements: List[Any]) -> set:
    """Normalized text of static elements spatially near ``field`` (its label row)."""

    fx1, fy1, fx2, fy2 = field.bbox
    fh = max(1, fy2 - fy1)
    near = set()
    for e in elements:
        if getattr(e, "editable", False) or not getattr(e, "text", None):
            continue
        x1, y1, x2, y2 = e.bbox
        horiz = not (x2 < fx1 or x1 > fx2)                 # overlaps horizontally
        vert = (abs(y1 - fy2) <= 2 * fh or abs(fy1 - y2) <= 2 * fh   # just below / above
                or (y1 >= fy1 - fh and y2 <= fy2 + fh))              # roughly same row
        if horiz and vert:
            near.add(_norm(e.text))
    near.discard("")
    return near


def _dedup_overlapping_values(elements: List[Any], contain_thresh: float = 0.7) -> None:
    """One-owner rule: if two editable boxes read the same value and one largely
    *contains* the other, the same on-screen text was assigned to both. Keep the
    value on the smaller (more specific) box and clear it from the larger.

    Bug 4: uses containment (intersection / smaller-box area), not IoU — IoU is
    near-zero when a real field sits inside a giant phantom strip, so it would
    never fire for its own use case."""

    def _area(bb):
        return max(0, bb[2] - bb[0]) * max(0, bb[3] - bb[1])

    def _inter(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        return max(0, ix2 - ix1) * max(0, iy2 - iy1)

    ed = [e for e in elements if getattr(e, "editable", False) and e.value]
    for i in range(len(ed)):
        for j in range(i + 1, len(ed)):
            a, b = ed[i], ed[j]
            if a.value is None or b.value is None:
                continue
            if _norm(a.value) != _norm(b.value):
                continue
            smaller = min(_area(a.bbox), _area(b.bbox))
            if smaller <= 0 or _inter(a.bbox, b.bbox) / smaller < contain_thresh:
                continue
            (a if _area(a.bbox) > _area(b.bbox) else b).value = None
