"""Engine 1 - VIDEO2ACTION (VideoAgentTrek-style).

A two-stage inverse-dynamics module over a frame stream: detect *when* GUI
actions happen (temporal grounding), then recognize *what* (coordinates, typed
text). Best when all you have is pixels over time. Outputs the shared
:class:`~demo2skill.video.schema.Trajectory`.
"""

from demo2skill.video.video2action.idm import (
    ActionContentRecognizer,
    ActionInterval,
    TemporalActionDetector,
    Video2Action,
)
from demo2skill.video.video2action.frames import Frame, Frames
from demo2skill.video.video2action.backends.scripted import ScriptedBackend

__all__ = [
    "Video2Action",
    "TemporalActionDetector",
    "ActionContentRecognizer",
    "ActionInterval",
    "Frame",
    "Frames",
    "ScriptedBackend",
]
