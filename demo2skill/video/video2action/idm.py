"""The inverse-dynamics module (IDM): video frames -> action trajectory.

Two stages, after VideoAgentTrek's VIDEO2ACTION:

1. :class:`TemporalActionDetector` - scans the frame stream and proposes
   :class:`ActionInterval`s: *when* an action happens, *what kind*, and a
   confidence. (The "video grounding model" in the paper.)
2. :class:`ActionContentRecognizer` - for each interval, reads the surrounding
   frames and fills the structured parameters - coordinates, typed text, and the
   semantic target. (The "action-content recognizer" in the paper.)

:class:`Video2Action` wires them together into a :class:`Trajectory`. Both
stages are Protocols, so a deterministic backend and a VLM backend are
interchangeable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from demo2skill.video.video2action.frames import Frames
from demo2skill.video.schema import Trajectory, VideoAction


@dataclass
class ActionInterval:
    """A detected action's temporal location and coarse type (stage-1 output)."""

    action_type: str
    start_ms: int
    end_ms: int
    frame_index: Optional[int] = None
    confidence: float = 1.0
    # Any parameters already known at detection time (e.g. parsed from an
    # on-screen keystroke/click overlay) the recognizer can trust or refine.
    hint: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class TemporalActionDetector(Protocol):
    def detect(self, frames: Frames) -> List[ActionInterval]:
        """Return action intervals in temporal order."""
        ...


@runtime_checkable
class ActionContentRecognizer(Protocol):
    def recognize(self, index: int, interval: ActionInterval, frames: Frames) -> VideoAction:
        """Return a fully-parameterized :class:`VideoAction` for ``interval``."""
        ...


class Video2Action:
    """Run the two-stage IDM over a frame stream."""

    def __init__(
        self,
        detector: TemporalActionDetector,
        recognizer: ActionContentRecognizer,
    ) -> None:
        self.detector = detector
        self.recognizer = recognizer

    def run(
        self,
        frames: Frames,
        *,
        video_id: str,
        source: Optional[str] = None,
    ) -> Trajectory:
        intervals = self.detector.detect(frames)
        actions: List[VideoAction] = []
        for i, interval in enumerate(intervals):
            action = self.recognizer.recognize(i, interval, frames)
            action.index = i
            if action.start_ms is None:
                action.start_ms = interval.start_ms
            if action.end_ms is None:
                action.end_ms = interval.end_ms
            actions.append(action)
        return Trajectory(
            video_id=video_id,
            actions=actions,
            source=source,
            fps=frames.fps,
            width=frames.width,
            height=frames.height,
        )
