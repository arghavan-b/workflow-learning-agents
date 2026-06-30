"""Engine 2 - State-diff inverse dynamics (recommended for GUI).

Recovers the action that *connects* two parsed screen states - visual change
only proposes candidate moments; element-level before/after state plus cursor
evidence determine the action. Outputs the shared
:class:`~demo2skill.video.schema.Trajectory`, plus a reusable UI state graph.
"""

from demo2skill.video.statediff.state import ScreenState, UIElement
from demo2skill.video.statediff.cursor import CursorSample, CursorTrack
from demo2skill.video.statediff.matching import match_states
from demo2skill.video.statediff.inverse_dynamics import (
    InferredAction,
    StateDiffIDM,
    StateTrajectoryBuilder,
    TransitionProposer,
)
from demo2skill.video.statediff.graph import UIStateGraph
from demo2skill.video.statediff.parser import (
    ScreenParser,
    ScreenParserClient,
    ScriptedScreenParser,
    VLMScreenParser,
    build_state,
    load_states,
    parse_frames,
)

__all__ = [
    "ScreenState",
    "UIElement",
    "CursorSample",
    "CursorTrack",
    "match_states",
    "InferredAction",
    "StateDiffIDM",
    "StateTrajectoryBuilder",
    "TransitionProposer",
    "UIStateGraph",
    # pixels->ScreenState front (pluggable)
    "ScreenParser",
    "ScreenParserClient",
    "ScriptedScreenParser",
    "VLMScreenParser",
    "build_state",
    "load_states",
    "parse_frames",
]
