"""OCR value-fill: read typed field text ScreenVLM leaves blank.

The fill logic is a pure function, so it's tested with a fake OCR and a blank
image — no tesseract needed. Only editable, value-less elements get filled;
everything else is untouched. ``make_ocr`` selection and the missing-dependency
path are also checked.
"""

from __future__ import annotations

import importlib
import unittest

from demo2skill.video.statediff.parser.ocr import make_ocr, ocr_fill_values
from demo2skill.video.statediff.state import UIElement


class FakeOCR:
    def __init__(self, text="Bug in login flow"):
        self.text = text
        self.crops = 0

    def read(self, image):
        self.crops += 1
        return self.text


def _img(w=800, h=400):
    from PIL import Image
    return Image.new("RGB", (w, h), "white")


class OcrFillTest(unittest.TestCase):
    def test_fills_only_editable_empty_fields(self):
        els = [
            UIElement(id="title", role="textbox", bbox=(20, 20, 500, 60)),      # editable, empty
            UIElement(id="btn", role="button", text="Create", bbox=(20, 80, 120, 110)),
            UIElement(id="lbl", role="text", text="Add a title", bbox=(20, 5, 120, 18)),
        ]
        ocr = FakeOCR()
        ocr_fill_values(els, _img(), ocr)

        self.assertEqual(els[0].value, "Bug in login flow")   # editable filled
        self.assertIsNone(els[1].value)                        # button untouched
        self.assertIsNone(els[2].value)                        # static text untouched
        self.assertEqual(ocr.crops, 1)                         # only one crop OCR'd

    def test_does_not_overwrite_existing_value(self):
        els = [UIElement(id="t", role="textbox", value="already here", bbox=(20, 20, 500, 60))]
        ocr = FakeOCR()
        ocr_fill_values(els, _img(), ocr)
        self.assertEqual(els[0].value, "already here")
        self.assertEqual(ocr.crops, 0)

    def test_skips_degenerate_boxes(self):
        els = [UIElement(id="t", role="textbox", bbox=(20, 20, 21, 21))]
        ocr = FakeOCR()
        ocr_fill_values(els, _img(), ocr)
        self.assertIsNone(els[0].value)
        self.assertEqual(ocr.crops, 0)


class MakeOcrTest(unittest.TestCase):
    def test_none_returns_none(self):
        self.assertIsNone(make_ocr("none"))
        self.assertIsNone(make_ocr(None))

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            make_ocr("magic")

    def test_tesseract_missing_dep_raises_helpful(self):
        if importlib.util.find_spec("pytesseract") is not None:
            self.skipTest("pytesseract installed; absence path not exercised")
        with self.assertRaises(SystemExit) as ctx:
            make_ocr("tesseract")
        self.assertIn("ocr", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
