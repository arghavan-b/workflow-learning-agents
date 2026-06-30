"""Match UI elements between two consecutive screen states.

Stable element tracks are what let actions be read as state transitions even when
the whole screen barely changes. For elements i in S_t and j in S_{t+1}:

    M_ij = w_iou * IoU(b_i, b_j)
         + w_text * sim(text_i, text_j)
         + w_type * 1[type_i == type_j]
         + w_label * sim(label_i, label_j)

Greedy assignment on M (cheap, good enough for GUI scales) yields matched pairs;
unmatched elements are ``appeared`` (only in S_{t+1}) or ``disappeared`` (only in
S_t). Those three sets are the substrate for inverse-dynamics inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from demo2skill.video.statediff.state import ScreenState, UIElement, iou, text_sim

W_IOU, W_TEXT, W_TYPE, W_LABEL = 0.5, 0.25, 0.15, 0.10
MATCH_THRESHOLD = 0.35


def match_score(a: UIElement, b: UIElement) -> float:
    return (
        W_IOU * iou(a.bbox, b.bbox)
        + W_TEXT * text_sim(a.text, b.text)
        + W_TYPE * (1.0 if a.role == b.role else 0.0)
        + W_LABEL * text_sim(a.label or "", b.label or "")
    )


@dataclass
class StateMatch:
    pairs: List[Tuple[UIElement, UIElement]] = field(default_factory=list)  # (before, after)
    appeared: List[UIElement] = field(default_factory=list)   # only in after
    disappeared: List[UIElement] = field(default_factory=list)  # only in before
    scores: Dict[str, float] = field(default_factory=dict)     # before.id -> score

    def after_of(self, before_id: str):
        for before, after in self.pairs:
            if before.id == before_id:
                return after
        return None


def match_states(s_before: ScreenState, s_after: ScreenState) -> StateMatch:
    candidates = []
    for a in s_before.elements:
        for b in s_after.elements:
            score = match_score(a, b)
            if score >= MATCH_THRESHOLD:
                candidates.append((score, a, b))
    candidates.sort(key=lambda c: c[0], reverse=True)

    used_before, used_after = set(), set()
    result = StateMatch()
    for score, a, b in candidates:
        if a.id in used_before or b.id in used_after:
            continue
        used_before.add(a.id)
        used_after.add(b.id)
        result.pairs.append((a, b))
        result.scores[a.id] = round(score, 3)

    result.disappeared = [a for a in s_before.elements if a.id not in used_before]
    result.appeared = [b for b in s_after.elements if b.id not in used_after]
    return result
