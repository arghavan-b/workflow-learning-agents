"""Trajectory data objects and the bridge to a normalize-ready raw trace.

A :class:`Trajectory` is the VIDEO2ACTION output: an ordered list of
:class:`VideoAction` steps, each carrying its temporal location, recognized
parameters (coordinates / typed text / keys), and a *semantic* target read from
the frame (the button caption or field label). :meth:`Trajectory.to_raw_trace`
emits exactly the ``trace.json`` shape :mod:`demo2skill.trace.normalize`
consumes, so a video flows into the existing induction pipeline unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

RAW_TRACE_SCHEMA_VERSION = "demo2skill.video_trace.v0"

# VIDEO2ACTION action space (a GUI / pyautogui-style vocabulary). ``navigate``
# and the text/click actions map cleanly onto the normalizer's semantics; the
# rest are carried through so a richer recognizer loses nothing.
CLICK_ACTIONS = {"click", "left_click", "double_click", "right_click", "middle_click"}
TEXT_ACTIONS = {"type"}
NAV_ACTIONS = {"navigate"}
ALL_ACTIONS = CLICK_ACTIONS | TEXT_ACTIONS | NAV_ACTIONS | {
    "key", "scroll", "drag", "move", "wait", "terminate",
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class VideoAction:
    """One recognized step in a video trajectory."""

    index: int
    action_type: str
    # temporal grounding (stage 1)
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    frame_index: Optional[int] = None
    # recognized content (stage 2)
    x: Optional[int] = None
    y: Optional[int] = None
    text: Optional[str] = None          # typed text
    keys: Optional[str] = None          # e.g. "ctrl+s", "enter"
    scroll_dx: Optional[int] = None
    scroll_dy: Optional[int] = None
    url: Optional[str] = None
    # semantic target read from the frame (bridges video -> induction)
    target_text: Optional[str] = None   # button / link caption
    target_label: Optional[str] = None  # field label
    target_role: Optional[str] = None   # textbox / button / link ...
    page_title: Optional[str] = None
    screenshot_path: Optional[str] = None
    confidence: float = 1.0

    def to_raw_event(self) -> Dict[str, Any]:
        """Serialize to a raw recorder-style event for the normalizer."""

        event_id = f"vid_{self.index:06d}"
        base: Dict[str, Any] = {
            "event_id": event_id,
            "timestamp": self.start_ms,
            "action_type": _raw_action_type(self.action_type),
            "url": self.url,
            "page_title": self.page_title,
            "screenshot_path": self.screenshot_path,
            "confidence": self.confidence,
        }

        if self.action_type in NAV_ACTIONS:
            return _drop_none(base)

        element = _drop_none({
            "role": self.target_role,
            "label": self.target_label,
            "text": self.target_text,
        })

        if self.action_type in TEXT_ACTIONS:
            base.update({
                "typed_text": self.text,
                "keyboard_text": self.text,
                "target_label": self.target_label,
            })
            element.setdefault("role", "textbox")
            if self.text is not None:
                element["value"] = self.text
        elif self.action_type in CLICK_ACTIONS:
            base.update({
                "target_text": self.target_text,
                "target_label": self.target_label,
                "mouse": _drop_none({"x": self.x, "y": self.y}) or None,
            })
        elif self.action_type == "key":
            base["keyboard_text"] = self.keys
        elif self.action_type == "scroll":
            base["scroll"] = _drop_none({"dx": self.scroll_dx, "dy": self.scroll_dy}) or None

        if element:
            base["element"] = element
        return _drop_none(base)


@dataclass
class Trajectory:
    video_id: str
    actions: List[VideoAction] = field(default_factory=list)
    source: Optional[str] = None          # file path or URL
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    created_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id,
            "source": self.source,
            "fps": self.fps,
            "resolution": _drop_none({"width": self.width, "height": self.height}) or None,
            "created_at": self.created_at,
            "actions": [_drop_none(a.__dict__) for a in self.actions],
        }

    def to_raw_trace(self) -> Dict[str, Any]:
        """A ``trace.json`` payload ready for :func:`trace.normalize.normalize_trace`."""

        return {
            "schema_version": RAW_TRACE_SCHEMA_VERSION,
            "created_at": self.created_at,
            "metadata": _drop_none({
                "source_modality": "video",
                "source": self.source,
                "video_id": self.video_id,
                "fps": self.fps,
                "width": self.width,
                "height": self.height,
            }),
            "events": [a.to_raw_event() for a in self.actions],
        }


def _raw_action_type(action_type: str) -> str:
    if action_type in NAV_ACTIONS:
        return "navigation"
    if action_type in TEXT_ACTIONS:
        return "type"
    if action_type in CLICK_ACTIONS:
        return "click"
    return action_type


def _drop_none(mapping: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in mapping.items() if v not in (None, "", [], {})}
