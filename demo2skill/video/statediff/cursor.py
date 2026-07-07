"""Cursor-centric evidence for disambiguating GUI actions.

In a screen recording the cursor is often the strongest action signal, and it is
essential for actions that leave little or no persistent state change (clicking
an already-active tab) and for resolving ambiguity (click vs keyboard shortcut
that produce the same effect). A click looks like:

    movement -> deceleration -> short dwell -> local UI response

This module tracks the pointer and exposes the evidence the inverse-dynamics
stage uses: where the cursor dwelled during a transition, and whether it showed a
click signature near a given element.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from demo2skill.video.statediff.state import UIElement


@dataclass
class CursorSample:
    ms: int
    x: int
    y: int
    visible: bool = True
    clicking: bool = False  # set if the recording exposes a click animation/down-state


class CursorTrack:
    def __init__(self, samples: List[CursorSample]) -> None:
        self.samples = sorted(samples, key=lambda s: s.ms)

    def __len__(self) -> int:
        return len(self.samples)

    def window(self, start_ms: int, end_ms: int) -> List[CursorSample]:
        return [s for s in self.samples if start_ms <= s.ms <= end_ms]

    def position_at(self, ms: int) -> Optional[Tuple[int, int]]:
        if not self.samples:
            return None
        s = min(self.samples, key=lambda s: abs(s.ms - ms))
        return (s.x, s.y)

    def dwell_point(self, start_ms: int, end_ms: int) -> Optional[Tuple[int, int]]:
        """Where the cursor settled in the window (lowest-velocity sample)."""

        win = self.window(start_ms, end_ms)
        if not win:
            return None
        if len(win) == 1:
            return (win[0].x, win[0].y)
        best, best_v = win[0], float("inf")
        for prev, cur in zip(win, win[1:]):
            v = abs(cur.x - prev.x) + abs(cur.y - prev.y)
            if v < best_v:
                best_v, best = v, cur
        return (best.x, best.y)

    def moved(self, start_ms: int, end_ms: int, eps: int = 4) -> bool:
        win = self.window(start_ms, end_ms)
        if len(win) < 2:
            return False
        span = max(abs(a.x - b.x) + abs(a.y - b.y) for a in win for b in win)
        return span > eps

    def click_signature(self, start_ms: int, end_ms: int) -> bool:
        """A click is movement that settles (or an explicit click flag)."""

        win = self.window(start_ms, end_ms)
        if any(s.clicking for s in win):
            return True
        return self.dwell_point(start_ms, end_ms) is not None

    def element_under(self, state, start_ms: int, end_ms: int,
                      radius: float = 40.0) -> Optional[UIElement]:
        point = self.dwell_point(start_ms, end_ms)
        if point is None:
            return None
        # Prefer a containing element; otherwise snap to the nearest within radius.
        return state.at(*point) or state.near(*point, radius=radius)

    @classmethod
    def from_records(cls, records) -> "CursorTrack":
        return cls([
            CursorSample(
                ms=int(r["ms"]), x=int(r["x"]), y=int(r["y"]),
                visible=bool(r.get("visible", True)),
                clicking=bool(r.get("clicking", False)),
            )
            for r in (records or [])
        ])

    @classmethod
    def empty(cls) -> "CursorTrack":
        return cls([])
