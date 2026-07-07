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
        els = [UIElement(id="t", role="textbox", bbox=(20, 20, 22, 22))]
        ocr = FakeOCR()
        ocr_fill_values(els, _img(), ocr)
        self.assertIsNone(els[0].value)
        self.assertEqual(ocr.crops, 0)

    def test_rejects_label_echo(self):
        # OCR reads the field's neighboring label instead of a typed value.
        els = [
            UIElement(id="lbl", role="text", text="Add a title", bbox=(20, 5, 200, 18)),
            UIElement(id="title", role="textbox", bbox=(20, 25, 500, 65)),
        ]
        ocr = FakeOCR("Add a title")           # echoes the label
        ocr_fill_values(els, _img(), ocr)
        self.assertIsNone(els[1].value)         # rejected, not stored as value

    def test_rejects_generic_placeholder(self):
        els = [UIElement(id="title", role="textbox", bbox=(20, 25, 500, 65))]
        ocr_fill_values(els, _img(), FakeOCR("Title"))   # a placeholder
        self.assertIsNone(els[0].value)

    def test_one_owner_dedup_on_overlapping_fields(self):
        # Two overlapping editable boxes reading the same value -> keep only the
        # smaller (more specific) owner.
        big = UIElement(id="big", role="textbox", bbox=(20, 20, 520, 60))
        small = UIElement(id="small", role="textbox", bbox=(22, 22, 500, 58))  # inside big
        ocr_fill_values([big, small], _img(), FakeOCR("Bug in login flow"))
        self.assertEqual(small.value, "Bug in login flow")   # specific box keeps it
        self.assertIsNone(big.value)                          # larger box cleared


class PaddleParseTest(unittest.TestCase):
    def test_join_handles_2x_list_shape(self):
        from demo2skill.video.statediff.parser.ocr import PaddleOCRBackend
        # PaddleOCR 2.x: [ [ [box, (text, conf)], ... ] ]
        result = [[
            [[[0, 0], [1, 0], [1, 1], [0, 1]], ("Bug in login", 0.99)],
            [[[0, 2], [1, 2], [1, 3], [0, 3]], ("flow", 0.98)],
        ]]
        self.assertEqual(PaddleOCRBackend._join(result), "Bug in login flow")

    def test_join_handles_3x_dict_shape(self):
        from demo2skill.video.statediff.parser.ocr import PaddleOCRBackend
        # PaddleOCR 3.x predict: list of dicts with rec_texts
        result = [{"rec_texts": ["Bug in login", "flow"], "rec_scores": [0.99, 0.98]}]
        self.assertEqual(PaddleOCRBackend._join(result), "Bug in login flow")

    def test_join_handles_empty(self):
        from demo2skill.video.statediff.parser.ocr import PaddleOCRBackend
        self.assertEqual(PaddleOCRBackend._join(None), "")
        self.assertEqual(PaddleOCRBackend._join([None]), "")

    def test_join_handles_3x_ocrresult_object(self):
        from demo2skill.video.statediff.parser.ocr import PaddleOCRBackend

        class OCRResult(dict):
            pass  # 3.x returns dict-like result objects

        result = [OCRResult(rec_texts=["Bug in login", "flow"], rec_scores=[0.99, 0.98])]
        self.assertEqual(PaddleOCRBackend._join(result), "Bug in login flow")

    def test_join_drops_low_confidence_and_cjk(self):
        from demo2skill.video.statediff.parser.ocr import PaddleOCRBackend
        result = [[
            [[[0, 0]], ("Bug in login", 0.99)],
            [[[0, 2]], ("garbage", 0.20)],       # low confidence -> dropped
            [[[0, 4]], ("六口二", 0.95)],           # CJK in EN session -> dropped
        ]]
        out = PaddleOCRBackend._join(result, drop_score=0.6, drop_cjk=True)
        self.assertEqual(out, "Bug in login")


class MakeOcrTest(unittest.TestCase):
    def test_none_returns_none(self):
        self.assertIsNone(make_ocr("none"))
        self.assertIsNone(make_ocr(None))

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            make_ocr("magic")

    def test_paddle_missing_dep_raises_helpful(self):
        import importlib
        if importlib.util.find_spec("paddleocr") is not None:
            self.skipTest("paddleocr installed; absence path not exercised")
        with self.assertRaises(SystemExit) as ctx:
            make_ocr("paddle")
        self.assertIn("paddleocr", str(ctx.exception))

    def test_tesseract_missing_dep_raises_helpful(self):
        if importlib.util.find_spec("pytesseract") is not None:
            self.skipTest("pytesseract installed; absence path not exercised")
        with self.assertRaises(SystemExit) as ctx:
            make_ocr("tesseract")
        self.assertIn("ocr", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
