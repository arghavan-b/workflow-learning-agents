"""Induce trajectories (and reusable skills) from raw tutorial videos.

Two interchangeable engines live in their own subpackages; both emit the shared
:class:`~demo2skill.video.schema.Trajectory`, which serializes to a raw
``trace.json`` and flows through ``trace.normalize`` -> ``induction``:

* :mod:`demo2skill.video.video2action` - VideoAgentTrek-style two-stage inverse
  dynamics over a frame stream (detect *when*, recognize *what*).
* :mod:`demo2skill.video.statediff` - state-diff inverse dynamics: recover the
  action connecting two parsed screen states (recommended for GUI).
"""

from demo2skill.video.schema import Trajectory, VideoAction

# Engine 1: VIDEO2ACTION (frame-based)
from demo2skill.video.video2action import (
    ActionContentRecognizer,
    ActionInterval,
    Frame,
    Frames,
    ScriptedBackend,
    TemporalActionDetector,
    Video2Action,
)

# Engine 2: state-diff inverse dynamics
from demo2skill.video.statediff import (
    CursorSample,
    CursorTrack,
    InferredAction,
    ScreenState,
    StateDiffIDM,
    StateTrajectoryBuilder,
    TransitionProposer,
    UIElement,
    UIStateGraph,
    match_states,
)

__all__ = [
    "Trajectory",
    "VideoAction",
    # engine 1
    "Video2Action",
    "TemporalActionDetector",
    "ActionContentRecognizer",
    "ActionInterval",
    "Frame",
    "Frames",
    "ScriptedBackend",
    # engine 2
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
]
