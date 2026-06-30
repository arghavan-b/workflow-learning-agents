import unittest

from demo2skill.trace.normalize import normalize_event, normalize_trace


class TraceNormalizeTests(unittest.TestCase):
    def test_type_event_becomes_fill_field_with_label_and_value(self):
        event = normalize_event(
            {
                "event_id": "evt_000002",
                "timestamp": "2026-05-28T10:30:00Z",
                "url": "https://example.com/form",
                "page_title": "Form",
                "action_type": "type",
                "selector": "input[name=\"amount\"]",
                "typed_text": "42.50",
                "element": {
                    "tag": "input",
                    "type": "number",
                    "label": "Amount",
                    "role": "textbox",
                    "name": "amount",
                },
            }
        )

        self.assertEqual(event["semantic_action"], "fill_field")
        self.assertEqual(event["target"]["label"], "Amount")
        self.assertEqual(event["target"]["role"], "textbox")
        self.assertEqual(event["value"], "42.50")
        self.assertEqual(event["page_context"], "Form (https://example.com/form)")

    def test_click_event_preserves_semantic_target(self):
        event = normalize_event(
            {
                "event_id": "evt_000003",
                "action_type": "click",
                "target_text": "New Issue",
                "selector": "a#new_issue_link",
                "element": {"tag": "a", "text": "New Issue"},
            }
        )

        self.assertEqual(event["semantic_action"], "click")
        self.assertEqual(event["target"]["text"], "New Issue")
        self.assertEqual(event["target"]["role"], "link")
        self.assertEqual(event["target"]["selector"], "a#new_issue_link")

    def test_change_on_file_input_becomes_upload_file(self):
        event = normalize_event(
            {
                "event_id": "evt_000004",
                "action_type": "change",
                "value": "receipt.pdf",
                "element": {"tag": "input", "type": "file", "label": "Receipt"},
            }
        )

        self.assertEqual(event["semantic_action"], "upload_file")
        self.assertEqual(event["target"]["semantic"], "file_upload")
        self.assertEqual(event["target"]["label"], "Receipt")

    def test_normalize_trace_deduplicates_exact_adjacent_noise(self):
        trace = normalize_trace(
            {
                "schema_version": "demo2skill.raw_trace.v0",
                "events": [
                    {"event_id": "evt_1", "action_type": "navigation", "url": "https://x.test"},
                    {"event_id": "evt_2", "action_type": "navigation", "url": "https://x.test"},
                    {
                        "event_id": "evt_3",
                        "action_type": "type",
                        "typed_text": "hello",
                        "element": {"tag": "input", "label": "Title"},
                    },
                ],
            }
        )

        self.assertEqual(trace["schema_version"], "demo2skill.semantic_trace.v0")
        self.assertEqual(len(trace["events"]), 2)
        self.assertEqual(trace["events"][0]["source_event_id"], "evt_2")


if __name__ == "__main__":
    unittest.main()
