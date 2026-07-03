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
from typing import Any, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger("demo2skill.parser")


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


def make_ocr(name: Optional[str]) -> Optional[OCR]:
    """Construct an OCR backend by name (``none`` -> ``None``)."""

    if not name or name == "none":
        return None
    if name == "tesseract":
        return TesseractOCR()
    if name == "easyocr":
        return EasyOCRBackend()
    raise ValueError(f"unknown OCR backend {name!r} (use 'tesseract', 'easyocr', or 'none')")


def ocr_fill_values(elements: List[Any], image: Any, ocr: OCR,
                    *, pad: int = 2, min_size: int = 4) -> List[Any]:
    """Fill each editable element's ``value`` by OCR-ing its crop of ``image``.

    ``image`` is the original full-resolution frame (better OCR than the
    downscaled model input). Only editable elements without a value are touched;
    everything else is left exactly as ScreenVLM produced it.
    """

    width, height = image.size
    for el in elements:
        if not getattr(el, "editable", False):
            continue
        if el.value and str(el.value).strip():
            continue
        x1, y1, x2, y2 = el.bbox
        if (x2 - x1) < min_size or (y2 - y1) < min_size:
            continue
        crop = image.crop((max(0, x1 - pad), max(0, y1 - pad),
                           min(width, x2 + pad), min(height, y2 + pad)))
        try:
            text = ocr.read(crop)
        except Exception as exc:  # a bad crop must not kill the whole run
            logger.warning("OCR failed on %s: %s", getattr(el, "id", "?"), exc)
            continue
        if text:
            el.value = text
            if not el.text:
                el.text = text
    return elements
