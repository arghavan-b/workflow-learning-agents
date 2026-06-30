import json
import tempfile
import unittest
from pathlib import Path

from demo2skill.recorder.event_logger import EventLogger, compact_element_info


class EventLoggerTests(unittest.TestCase):
    def test_append_assigns_stable_event_ids_and_writes_trace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = EventLogger(Path(tmpdir), metadata={"start_url": "https://example.com"})

            first = logger.append({"action_type": "click"})
            second = logger.append({"action_type": "type", "typed_text": "hello"})
            trace_path = logger.save()

            self.assertEqual(first["event_id"], "evt_000001")
            self.assertEqual(second["event_id"], "evt_000002")
            data = json.loads(trace_path.read_text(encoding="utf-8"))
            self.assertEqual(data["schema_version"], "demo2skill.raw_trace.v0")
            self.assertEqual(data["metadata"]["start_url"], "https://example.com")
            self.assertEqual(len(data["events"]), 2)

    def test_compact_element_info_drops_empty_values(self):
        compacted = compact_element_info(
            {"selector": "button", "label": "", "role": "button", "empty": None}
        )
        self.assertEqual(compacted, {"selector": "button", "role": "button"})


if __name__ == "__main__":
    unittest.main()

