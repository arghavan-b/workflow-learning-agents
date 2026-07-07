"""Engine 2 - State-diff inverse dynamics (recommended for GUI).

Recovers the action that *connects* two parsed screen states - visual change
only proposes candidate moments; element-level before/after state plus cursor
evidence determine the action. Outputs the shared
:class:`~demo2skill.video.schema.Trajectory`, plus a reusable UI state graph.
"""

from demo2skill.video.statediff.state import ScreenState, UIElement
from demo2skill.video.statediff.cursor import CursorSample, CursorTrack
from demo2skill.video.statediff.cursor_detect import (
    CursorDetector,
    CursorHit,
    TemplateCursorDetector,
    build_cursor_detector,
    detect_cursor_track,
)
from demo2skill.video.statediff.matching import match_states
from demo2skill.video.statediff.inverse_dynamics import (
    InferredAction,
    StateDiffIDM,
    StateTrajectoryBuilder,
    TransitionProposer,
)
from demo2skill.video.statediff.graph import UIStateGraph
from demo2skill.video.statediff.field_text import (
    classify_field_text,
    chain_editables,
    vote_value,
)
from demo2skill.video.statediff.parser import (
    ScreenParser,
    ScreenParserClient,
    ScreenVLMParser,
    ScriptedScreenParser,
    VLMScreenParser,
    build_state,
    load_states,
    parse_frames,
    parse_screentag,
    state_to_dict,
    states_payload,
)

__all__ = [
    "ScreenState",
    "UIElement",
    "CursorSample",
    "CursorTrack",
    "CursorDetector",
    "CursorHit",
    "TemplateCursorDetector",
    "build_cursor_detector",
    "detect_cursor_track",
    "match_states",
    "classify_field_text",
    "chain_editables",
    "vote_value",
    "InferredAction",
    "StateDiffIDM",
    "StateTrajectoryBuilder",
    "TransitionProposer",
    "UIStateGraph",
    # pixels->ScreenState front (pluggable)
    "ScreenParser",
    "ScreenParserClient",
    "ScreenVLMParser",
    "ScriptedScreenParser",
    "VLMScreenParser",
    "parse_screentag",
    "build_state",
    "load_states",
    "parse_frames",
    "state_to_dict",
    "states_payload",
]
