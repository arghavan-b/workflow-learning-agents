"""The pixels->ScreenState front: fill the pluggable parser slot statediff assumes.

These tests prove the seam works end to end - a parser turns frames into
``ScreenState``s, and the existing state-diff IDM recovers the same actions it
did from hand-authored states. Both a model-free replay parser and a stubbed
ScreenVLM client are exercised, so the whole ``frames -> states -> actions``
chain runs with no real model.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import List

from demo2skill.video.statediff import (
    ScreenParser,
    ScriptedScreenParser,
    StateTrajectoryBuilder,
    UIStateGraph,
    VLMScreenParser,
    build_state,
    load_states,
    parse_frames,
)
from demo2skill.video.statediff.parser.vlm import ScreenParserClient
from demo2skill.video.video2action.frames import Frame, Frames

EXAMPLE = Path(__file__).resolve().parents[1] / "demo2skill" / "examples" / "github_issue"


def _example():
    return json.loads((EXAMPLE / "screen_states.json").read_text())


def _frames_for(states) -> Frames:
    return Frames([Frame(index=s["index"], ms=s["ms"]) for s in states])


class StubScreenVLM:
    """A ScreenVLM stand-in: returns the example's parsed JSON, one per call."""

    def __init__(self, states: List[dict]) -> None:
        self._states = states
        self.calls = 0

    def complete(self, *, system: str, prompt: str, images) -> str:
        data = self._states[self.calls]
        self.calls += 1
        return json.dumps(data)


class VLMRobustnessTest(unittest.TestCase):
    def test_parse_json_strips_fences_and_prose(self):
        from demo2skill.video.statediff.parser.vlm import _parse_json
        wrapped = 'Here is the parse:\n```json\n{"elements": [{"role": "button"}]}\n```\nDone.'
        self.assertEqual(_parse_json(wrapped), {"elements": [{"role": "button"}]})

    def test_bad_response_yields_empty_state_not_crash(self):
        from demo2skill.video.statediff.parser.vlm import VLMScreenParser

        class BadClient:
            def complete(self, *, system, prompt, images):
                return '{"elements": [ {"role": "button"  <<truncated'   # invalid JSON

        state = VLMScreenParser(BadClient()).parse(b"x", index=5, ms=500)
        self.assertEqual(state.index, 5)
        self.assertEqual(state.elements, [])        # empty, but no exception

    def test_bare_array_response_is_wrapped(self):
        from demo2skill.video.statediff.parser.vlm import VLMScreenParser

        class ArrayClient:
            def complete(self, *, system, prompt, images):
                return '[{"role": "textbox", "text": "hi", "bbox": [0,0,10,10]}]'

        state = VLMScreenParser(ArrayClient()).parse(b"x", index=0, ms=0)
        self.assertEqual(len(state.elements), 1)
        self.assertEqual(state.elements[0].role, "textbox")


class InvertImageTest(unittest.TestCase):
    def test_invert_flips_colors(self):
        import io
        from PIL import Image
        from demo2skill.video.statediff.parser.base import invert_image
        buf = io.BytesIO()
        Image.new("RGB", (4, 4), (10, 20, 30)).save(buf, format="PNG")
        out = invert_image(buf.getvalue())
        px = Image.open(io.BytesIO(out)).convert("RGB").getpixel((0, 0))
        self.assertEqual(px, (245, 235, 225))   # 255 - original


class BuildStateTest(unittest.TestCase):
    def test_build_state_is_tolerant(self):
        state = build_state(
            {"url": "u", "title": "t",
             "elements": [{"role": "button", "text": "Go"}]},  # no id / bbox
            index=3, ms=900,
        )
        self.assertEqual(state.index, 3)
        self.assertEqual(state.ms, 900)
        el = state.elements[0]
        self.assertTrue(el.id)                 # synthesized
        self.assertEqual(el.bbox, (0, 0, 0, 0))  # defaulted
        self.assertEqual(el.role, "button")


class LoadStatesParityTest(unittest.TestCase):
    def test_loader_matches_handwritten_path(self):
        states, cursor = load_states(_example())
        traj = StateTrajectoryBuilder(cursor).build(states, video_id="issue")
        self.assertEqual([a.action_type for a in traj.actions],
                         ["click", "click", "type", "type"])
        graph = UIStateGraph.build(states, cursor)
        self.assertEqual(len(graph.nodes), 5)
        self.assertEqual(len(graph.edges), 4)


class ScriptedParserTest(unittest.TestCase):
    def test_parse_frames_replays_states(self):
        data = _example()
        parser = ScriptedScreenParser(data["states"])
        self.assertIsInstance(parser, ScreenParser)  # satisfies the protocol

        states = parse_frames(parser, _frames_for(data["states"]))
        _, cursor = load_states(data)
        traj = StateTrajectoryBuilder(cursor).build(states, video_id="issue")
        self.assertEqual([a.action_type for a in traj.actions],
                         ["click", "click", "type", "type"])


class VLMParserTest(unittest.TestCase):
    def test_stub_screenvlm_drives_full_chain(self):
        data = _example()
        client = StubScreenVLM(data["states"])
        self.assertIsInstance(client, ScreenParserClient)
        parser = VLMScreenParser(client)
        self.assertIsInstance(parser, ScreenParser)

        states = parse_frames(parser, _frames_for(data["states"]))
        self.assertEqual(client.calls, len(data["states"]))  # one VLM call per frame

        _, cursor = load_states(data)
        traj = StateTrajectoryBuilder(cursor).build(states, video_id="issue")

        kinds = [a.action_type for a in traj.actions]
        self.assertEqual(kinds, ["click", "click", "type", "type"])
        types = [a for a in traj.actions if a.action_type == "type"]
        self.assertEqual(types[0].target_label, "Title")
        self.assertEqual(types[0].text, "Bug in login flow")


if __name__ == "__main__":
    unittest.main()
