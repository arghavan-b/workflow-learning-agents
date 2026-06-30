"""Deterministic backend driven by explicit action records.

This is the no-model path. Its input is a list of action dicts that can come
from sources a tutorial often *already* contains: on-screen keystroke/click
overlays (e.g. KeyCast), creator-provided chapter markers, or a hand-authored
sidecar for testing. It implements both IDM stages, so the full
video -> trajectory -> skill chain runs with zero ML and is unit-testable.

Each record is a dict like::

    {"action": "type", "label": "Title", "text": "Bug in login flow",
     "start_ms": 8000, "end_ms": 9500}
    {"action": "click", "target_text": "New issue", "x": 410, "y": 220}
    {"action": "navigate", "url": "https://github.com/owner/repo/issues/new",
     "page_title": "New Issue"}
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

from demo2skill.video.video2action.frames import Frames
from demo2skill.video.video2action.idm import ActionInterval
from demo2skill.video.schema import VideoAction


class ScriptedBackend:
    """Implements ``TemporalActionDetector`` + ``ActionContentRecognizer``."""

    def __init__(self, records: List[Mapping[str, Any]], default_step_ms: int = 1500) -> None:
        self.records = list(records)
        self.default_step_ms = default_step_ms

    # -- stage 1: temporal detection ----------------------------------------
    def detect(self, frames: Frames) -> List[ActionInterval]:
        intervals: List[ActionInterval] = []
        cursor = 0
        for rec in self.records:
            start = int(rec.get("start_ms", cursor))
            end = int(rec.get("end_ms", start + self.default_step_ms))
            intervals.append(
                ActionInterval(
                    action_type=str(rec.get("action", "click")),
                    start_ms=start,
                    end_ms=end,
                    frame_index=rec.get("frame_index"),
                    confidence=float(rec.get("confidence", 1.0)),
                    hint=dict(rec),
                )
            )
            cursor = end
        return intervals

    # -- stage 2: content recognition ---------------------------------------
    def recognize(self, index: int, interval: ActionInterval, frames: Frames) -> VideoAction:
        rec: Dict[str, Any] = interval.hint
        return VideoAction(
            index=index,
            action_type=interval.action_type,
            start_ms=interval.start_ms,
            end_ms=interval.end_ms,
            frame_index=interval.frame_index,
            x=rec.get("x"),
            y=rec.get("y"),
            text=rec.get("text"),
            keys=rec.get("keys"),
            scroll_dx=rec.get("scroll_dx"),
            scroll_dy=rec.get("scroll_dy"),
            url=rec.get("url"),
            target_text=rec.get("target_text"),
            target_label=rec.get("label") or rec.get("target_label"),
            target_role=rec.get("role") or rec.get("target_role"),
            page_title=rec.get("page_title"),
            screenshot_path=rec.get("screenshot_path"),
            confidence=interval.confidence,
        )
