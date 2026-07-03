"""The real ScreenVLM path: ScreenTag markup -> ScreenState, and import-safety.

The model itself needs torch/transformers + a checkpoint, so ``ScreenVLMParser``
is only checked for its clean install error when the deps are absent. The
ScreenTag parser is a pure function and is tested directly on the model card's
example output, including coordinate rescaling and role mapping.
"""

from __future__ import annotations

import importlib
import unittest

from demo2skill.video.statediff.parser import parse_screentag
from demo2skill.video.statediff.parser import screenvlm as sv

SAMPLE = (
    "<screentag>\n"
    "<button><loc_10><loc_20><loc_50><loc_35>Submit</button>\n"
    "<link><loc_100><loc_200><loc_180><loc_210>Learn more</link>\n"
    "<navigation_bar><loc_0><loc_0><loc_500><loc_30>\n"
    "  <link><loc_10><loc_5><loc_60><loc_25>Home</link>\n"
    "  <input><loc_200><loc_5><loc_300><loc_25>search here</input>\n"
    "</navigation_bar>\n"
    "</screentag>"
)


class ParseScreenTagTest(unittest.TestCase):
    def test_extracts_all_elements_in_order(self):
        els = parse_screentag(SAMPLE, width=1000, height=500)
        roles = [e.role for e in els]
        # navigation_bar has no mapping -> keeps its lowercased tag name
        self.assertEqual(roles, ["button", "link", "navigation_bar", "link", "textbox"])

    def test_coordinates_rescale_from_500_grid_to_pixels(self):
        els = parse_screentag(SAMPLE, width=1000, height=500)
        submit = els[0]
        # loc 10,20,50,35 on a 500 grid -> (20,20,100,35) at 1000x500
        self.assertEqual(submit.bbox, (20, 20, 100, 35))

    def test_input_text_becomes_value_for_editable(self):
        els = parse_screentag(SAMPLE, width=1000, height=500)
        field = els[-1]
        self.assertEqual(field.role, "textbox")
        self.assertTrue(field.editable)
        self.assertEqual(field.value, "search here")   # surfaced for typing detection
        button = els[0]
        self.assertIsNone(button.value)                 # non-editable: no value

    def test_role_mapping_covers_common_controls(self):
        self.assertEqual(sv.map_role("Checkbox"), "checkbox")
        self.assertEqual(sv.map_role("dropdown"), "combobox")
        self.assertEqual(sv.map_role("search"), "searchbox")
        self.assertEqual(sv.map_role("unknown_tag"), "unknown_tag")


class ImportSafetyTest(unittest.TestCase):
    def test_missing_deps_raise_helpful_message(self):
        if (importlib.util.find_spec("torch") is not None
                and importlib.util.find_spec("transformers") is not None):
            self.skipTest("torch/transformers installed; absence path not exercised")
        with self.assertRaises(SystemExit) as ctx:
            sv.ScreenVLMParser()
        self.assertIn("screenvlm", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
