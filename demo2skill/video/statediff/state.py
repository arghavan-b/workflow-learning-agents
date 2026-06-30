"""Parsed screen states - the observation the inverse-dynamics module reasons over.

GUI actions are sparse, localized, and stateful: a checkbox flips in a 20x20
region, one character appears in a field, focus moves between inputs. Whole-frame
visual similarity misses these. So instead of representing a frame as one
embedding, we represent it as a *set of UI elements* with positions, text, and
state - a ``ScreenState`` (what a screen parser such as OmniParser / ScreenParse
emits). Actions are then recovered as transitions between element-level states.

The pixels->ScreenState step (OCR + element detection) is the pluggable front of
the pipeline; everything in this module operates on the parsed structure so it is
deterministic and testable without a model.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

BBox = Tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass
class UIElement:
    id: str
    role: str                       # textbox | button | link | checkbox | radio | combobox | option | menu | tab | dialog ...
    text: str = ""
    bbox: BBox = (0, 0, 0, 0)
    value: Optional[str] = None     # current field value (for editable elements)
    label: Optional[str] = None
    focused: bool = False
    checked: Optional[bool] = None
    selected: Optional[bool] = None

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) // 2, (y1 + y2) // 2

    @property
    def editable(self) -> bool:
        return self.role in {"textbox", "combobox", "searchbox"}

    def contains(self, x: int, y: int) -> bool:
        x1, y1, x2, y2 = self.bbox
        return x1 <= x <= x2 and y1 <= y <= y2

    def display(self) -> str:
        return self.label or self.text or self.id


@dataclass
class ScreenState:
    index: int
    ms: int
    elements: List[UIElement] = field(default_factory=list)
    url: Optional[str] = None
    title: Optional[str] = None

    def by_id(self, element_id: str) -> Optional[UIElement]:
        return next((e for e in self.elements if e.id == element_id), None)

    @property
    def element_ids(self) -> set:
        return {e.id for e in self.elements}

    @property
    def text_blob(self) -> str:
        return " ".join(sorted(f"{e.role}:{e.text}:{e.value or ''}" for e in self.elements))

    def at(self, x: int, y: int) -> Optional[UIElement]:
        """Smallest element whose box contains the point (topmost control)."""

        hits = [e for e in self.elements if e.contains(x, y)]
        if not hits:
            return None
        return min(hits, key=lambda e: _area(e.bbox))


# -- distances used for state dedup (sameState) ----------------------------

def _area(b: BBox) -> int:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def iou(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = _area(a) + _area(b) - inter
    return inter / union if union else 0.0


def text_sim(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a or "", b or "").ratio()


def layout_distance(s1: ScreenState, s2: ScreenState) -> float:
    """1 - mean best-IoU of elements across the two states (0 = same layout)."""

    if not s1.elements and not s2.elements:
        return 0.0
    if not s1.elements or not s2.elements:
        return 1.0
    total = 0.0
    for e1 in s1.elements:
        total += max(iou(e1.bbox, e2.bbox) for e2 in s2.elements)
    return 1.0 - total / len(s1.elements)


def element_churn(s1: ScreenState, s2: ScreenState) -> float:
    """Fraction of element identities that changed (symmetric difference)."""

    a, b = s1.element_ids, s2.element_ids
    if not (a | b):
        return 0.0
    return len(a ^ b) / len(a | b)


def same_state(s1: ScreenState, s2: ScreenState,
               *, layout_tol: float = 0.05, text_tol: float = 0.02,
               churn_tol: float = 0.0) -> bool:
    """True if two screens are the same GUI state despite transient noise
    (blinking cursor, clock, pointer position) - used to dedup graph nodes."""

    return (
        layout_distance(s1, s2) <= layout_tol
        and (1.0 - text_sim(s1.text_blob, s2.text_blob)) <= text_tol
        and element_churn(s1, s2) <= churn_tol
    )
