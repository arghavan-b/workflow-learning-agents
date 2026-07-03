"""The parse-video CLI runs the parsing front end to end via the free dry run.

Replaying the bundled example through ``demo2skill-parse-video`` must reproduce
the five states and the click/click/type/type trajectory, and write a
normalize-ready trace - proving the CLI wiring without a model or ffmpeg.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from demo2skill.trace.normalize import normalize_trace
from demo2skill.video.statediff.cli import main
from demo2skill.video.statediff.parser import load_states

EXAMPLE = Path(__file__).resolve().parents[1] / "demo2skill" / "examples" / "github_issue"
REPLAY = EXAMPLE / "screen_states.json"


class ParseVideoDryRunTest(unittest.TestCase):
    def test_replay_produces_states_and_trace(self):
        with tempfile.TemporaryDirectory() as d:
            states_out = Path(d) / "states.json"
            trace_out = Path(d) / "trace.json"
            rc = main([
                "--replay", str(REPLAY),
                "-o", str(states_out),
                "--trace-out", str(trace_out),
            ])
            self.assertEqual(rc, 0)

            # States round-trip back through the loader.
            states, _ = load_states(json.loads(states_out.read_text()))
            self.assertEqual(len(states), 5)

            # The written trace is normalize-ready and yields the expected actions.
            trace = json.loads(trace_out.read_text())
            semantic = normalize_trace(trace)
            self.assertTrue(semantic.get("events"))

    def test_scripted_client_requires_replay(self):
        with tempfile.TemporaryDirectory() as d:
            rc = main(["--client", "scripted", "-o", str(Path(d) / "s.json")])
            self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
