"""The pixels->ScreenState front: the pluggable parser slot statediff assumes.

``statediff`` recovers actions by diffing *parsed* screen states; it never reads
a pixel. This module defines the contract for the component that produces those
states from frames, plus the driver that runs it over a frame stream:

    frames --[ ScreenParser ]--> ScreenState[] --> statediff (proposer/IDM/graph)

A ``ScreenParser`` turns one frame into one :class:`ScreenState` (every visible
element, not just the task-relevant one - dense parsing is what lets the matcher
track elements across frames). Two adapters implement it: a ScreenVLM-style VLM
backend (:mod:`demo2skill.video.statediff.parser.vlm`) and the model-free
:class:`ScriptedScreenParser` here, which replays pre-parsed states so the whole
chain is testable without a model. ``build_state`` / ``load_states`` are the
shared dict->ScreenState constructors both paths use.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from demo2skill.video.statediff.cursor import CursorTrack
from demo2skill.video.statediff.state import ScreenState, UIElement


@runtime_checkable
class ScreenParser(Protocol):
    """Turn a single frame into a parsed :class:`ScreenState`.

    ``image`` is the frame bytes (PNG/JPEG) or ``None`` for pixel-free replay.
    ``index`` and ``ms`` position the state in the stream.
    """

    def parse(self, image: Optional[bytes], *, index: int, ms: int) -> ScreenState:
        ...


def build_element(data: Dict[str, Any], fallback_index: int = 0) -> UIElement:
    """Construct a :class:`UIElement` from a parsed element dict (tolerant)."""

    return UIElement(
        id=str(data.get("id") or f"el_{fallback_index:03d}"),
        role=str(data.get("role") or "text"),
        text=str(data.get("text") or ""),
        bbox=_bbox(data.get("bbox")),
        value=data.get("value"),
        label=data.get("label"),
        focused=bool(data.get("focused", False)),
        checked=data.get("checked"),
        selected=data.get("selected"),
    )


def build_state(data: Dict[str, Any], *, index: Optional[int] = None,
                ms: Optional[int] = None) -> ScreenState:
    """Construct a :class:`ScreenState` from a parsed-screen dict.

    The dict shape is the dense parser contract: ``{url, title, elements:[...]}``
    where each element carries ``id, role, bbox, text, value, label, focused,
    checked, selected``. ``index`` / ``ms`` override the dict's own values so a
    parser can stamp them from the frame.
    """

    idx = data.get("index", 0) if index is None else index
    when = data.get("ms", 0) if ms is None else ms
    elements = [build_element(e, i) for i, e in enumerate(data.get("elements", []))]
    return ScreenState(
        index=int(idx or 0),
        ms=int(when or 0),
        url=data.get("url"),
        title=data.get("title"),
        elements=elements,
    )


def load_states(data: Dict[str, Any]) -> Tuple[List[ScreenState], CursorTrack]:
    """Load the ``{cursor:[...], states:[...]}`` fixture shape into the objects
    statediff consumes. This is the deterministic, pixel-free path."""

    states = [build_state(s) for s in data.get("states", [])]
    states.sort(key=lambda s: s.index)
    return states, CursorTrack.from_records(data.get("cursor"))


def parse_frames(parser: ScreenParser, frames: Any) -> List[ScreenState]:
    """Run ``parser`` over a :class:`~demo2skill.video.video2action.frames.Frames`
    stream, yielding one :class:`ScreenState` per frame in temporal order.

    The proposer downstream collapses consecutive identical states, so the parser
    need not deduplicate - it just parses every frame it is given.
    """

    states: List[ScreenState] = []
    for frame in frames.frames:
        image = frame.bytes() if hasattr(frame, "bytes") else None
        states.append(parser.parse(image, index=frame.index, ms=frame.ms))
    return states


class ScriptedScreenParser:
    """A model-free parser that replays pre-parsed states (for tests / fixtures).

    Initialize with parsed-screen dicts or ready :class:`ScreenState`s keyed by
    frame index. ``parse`` ignores the image and returns the matching state, so
    the full ``parse_frames -> statediff`` chain runs without any VLM.
    """

    def __init__(self, states: Any) -> None:
        self._by_index: Dict[int, ScreenState] = {}
        for i, s in enumerate(states):
            state = s if isinstance(s, ScreenState) else build_state(s)
            key = state.index if isinstance(s, ScreenState) else s.get("index", i)
            self._by_index[int(key)] = state

    def parse(self, image: Optional[bytes], *, index: int, ms: int) -> ScreenState:
        state = self._by_index.get(index)
        if state is None:
            return ScreenState(index=index, ms=ms, elements=[])
        # Stamp the frame's clock so timing stays consistent with the stream.
        state.ms = ms
        return state


def _bbox(value: Any) -> Tuple[int, int, int, int]:
    if not value or len(value) != 4:
        return (0, 0, 0, 0)
    return (int(value[0]), int(value[1]), int(value[2]), int(value[3]))
