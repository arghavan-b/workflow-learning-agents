"""State-diff inverse dynamics: (O_t, O_{t+1}) -> action -> element -> effect.

The central formulation for GUI demonstrations. Rather than treating a
whole-frame visual discontinuity as an event boundary, we recover the action
that *connects* two parsed screen states, grounded in the element that changed
and disambiguated by cursor evidence:

    (S_t, S_{t+1}) --IDM--> a_t (type/target/args) --> effect Delta s_t

Visual change only *proposes* candidate moments (``TransitionProposer``);
element-level before/after state plus the cursor *determine* the action
(``StateDiffIDM``). Different actions can produce similar screen changes (click
Submit vs press Enter), so the cursor/keyboard evidence is what resolves them.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from demo2skill.video.statediff.cursor import CursorTrack
from demo2skill.video.statediff.matching import match_states
from demo2skill.video.schema import Trajectory, VideoAction
from demo2skill.video.statediff.state import ScreenState, UIElement, element_churn, same_state

NAV_CHURN = 0.6   # fraction of element identities replaced => page transition
SCROLL_DY = 20    # px coherent vertical shift => scroll


def _value(e) -> str:
    """An element's current content: its field value if set, else its text.

    Placeholder-aware: if the element's text *is* its placeholder, it is not
    content, so don't let it leak in as a value (Bug 1)."""

    if e.value not in (None, ""):
        return e.value
    ph = getattr(e, "placeholder_text", None)
    if ph and (e.text or "") == ph:
        return ""
    return e.text or ""


def _changed_boxes(match, before, after) -> list:
    """Bounding boxes of elements that appeared, disappeared, or changed state."""

    boxes = [e.bbox for e in match.appeared] + [e.bbox for e in match.disappeared]
    for b, a in match.pairs:
        # Content/state changes only. Role flips (Issues tab<->button parser
        # noise) are NOT treated as a change (Bug 3).
        if (_value(a) != _value(b) or a.checked != b.checked
                or a.selected != b.selected):
            boxes.append(a.bbox)
    return boxes


def _near_changed(point, match, before, after, pad: int = 25) -> bool:
    """Is ``point`` inside (or within ``pad`` of) any element that changed?"""

    px, py = point
    for x1, y1, x2, y2 in _changed_boxes(match, before, after):
        if (x1 - pad) <= px <= (x2 + pad) and (y1 - pad) <= py <= (y2 + pad):
            return True
    return False


@dataclass
class InferredAction:
    """A structured action recovered from a single state transition."""

    action_type: str                      # type | click | scroll | key | navigate | noop
    target: Optional[UIElement] = None
    args: Dict[str, Any] = field(default_factory=dict)
    precondition: Dict[str, Any] = field(default_factory=dict)
    effect: Dict[str, Any] = field(default_factory=dict)
    x: Optional[int] = None
    y: Optional[int] = None
    url: Optional[str] = None
    page_title: Optional[str] = None
    confidence: float = 1.0

    def to_video_action(self, index: int, start_ms: int, end_ms: int) -> VideoAction:
        t = self.target
        kind = "type" if self.action_type == "type" else (
            "navigate" if self.action_type == "navigate" else (
                self.action_type if self.action_type in {"scroll", "key"} else "click"))
        return VideoAction(
            index=index,
            action_type=kind,
            start_ms=start_ms,
            end_ms=end_ms,
            x=self.x, y=self.y,
            text=self.args.get("text"),
            keys=self.args.get("keys"),
            scroll_dy=self.args.get("dy"),
            url=self.url,
            target_text=(t.text or None) if t else None,
            target_label=(t.label or t.text or None) if t else None,
            target_role=t.role if t else None,
            page_title=self.page_title,
            confidence=self.confidence,
        )


@dataclass
class Transition:
    index: int
    before: ScreenState
    after: ScreenState


class TransitionProposer:
    """Stage 1: visual change proposes candidate moments.

    Here 'visual change' is computed at the element level (the parsed-state
    delta), which is far more sensitive to small GUI actions than global frame
    similarity. Consecutive states that are the *same* GUI state are skipped.
    """

    def propose(self, states: List[ScreenState]) -> List[Transition]:
        out: List[Transition] = []
        for i in range(len(states) - 1):
            before, after = states[i], states[i + 1]
            if same_state(before, after):
                continue
            out.append(Transition(index=i, before=before, after=after))
        return out


class StateDiffIDM:
    """Stage 3: determine the action from element-level diff + cursor evidence."""

    def __init__(self, cursor: Optional[CursorTrack] = None) -> None:
        self.cursor = cursor or CursorTrack.empty()

    def infer(self, before: ScreenState, after: ScreenState) -> InferredAction:
        match = match_states(before, after)
        pre = {"screen": after.title or after.url}
        common = dict(url=after.url, page_title=after.title, precondition=pre)
        cursor_pt = self.cursor.dwell_point(before.ms, after.ms)
        clicked = self.cursor.click_signature(before.ms, after.ms)
        cursor_moved = self.cursor.moved(before.ms, after.ms)

        # 1. Text entry: a matched editable element's value grew/changed.
        for b, a in match.pairs:
            if a.editable and _value(a) != _value(b) and _value(a):
                x, y = a.center
                return InferredAction(
                    "type", target=a, args={"text": _value(a)},
                    effect={"field_value": _value(a)},
                    x=x, y=y, confidence=0.95, **common)

        # 2. Toggle / checkbox / radio.
        for b, a in match.pairs:
            if a.role in {"checkbox", "radio"} and a.checked != b.checked:
                x, y = a.center
                return InferredAction(
                    "click", target=a, args={"set_checked": a.checked},
                    effect={"checked": a.checked}, x=x, y=y, confidence=0.9, **common)

        # 3. Selection (option/tab/list item became selected).
        for b, a in match.pairs:
            if a.selected and not b.selected:
                x, y = a.center
                return InferredAction(
                    "click", target=a, effect={"selected": True},
                    x=x, y=y, confidence=0.85, **common)

        # 4. Page transition: url changed or most elements turned over.
        if (after.url and after.url != before.url) or element_churn(before, after) >= NAV_CHURN:
            control = self.cursor.element_under(before, before.ms, after.ms)
            if clicked and control is not None and control.role in {"link", "button", "tab"}:
                # The most likely cause of the new screen is clicking that control.
                x, y = control.center
                return InferredAction(
                    "click", target=control, effect={"navigated_to": after.url},
                    x=x, y=y, confidence=0.8, **common)
            return InferredAction(
                "navigate", args={}, effect={"navigated_to": after.url},
                confidence=0.7, **common)

        # 5. Menu / dialog appeared near the cursor.
        opened = [e for e in match.appeared if e.role in {"menu", "dialog", "listbox", "option"}]
        if opened:
            control = self.cursor.element_under(before, before.ms, after.ms)
            act = "click"
            if control is not None:
                x, y = control.center
                return InferredAction(
                    act, target=control, effect={"opened": opened[0].role},
                    x=x, y=y, confidence=0.75, **common)

        # 6. Scroll: coherent vertical shift of matched content.
        dys = [a.center[1] - b.center[1] for b, a in match.pairs]
        if len(dys) >= 3 and abs(statistics.median(dys)) >= SCROLL_DY and not cursor_moved:
            dy = int(statistics.median(dys))
            return InferredAction("scroll", args={"dy": dy},
                                  effect={"viewport_dy": dy}, confidence=0.7, **common)

        # 7. Keyboard focus move (focus changed, cursor did not move/click).
        focus_after = next((e for e in after.elements if e.focused), None)
        focus_before = next((e for e in before.elements if e.focused), None)
        if focus_after and (not focus_before or focus_after.id != getattr(focus_before, "id", None)) \
                and not cursor_moved and not clicked:
            x, y = focus_after.center
            return InferredAction("key", target=focus_after, args={"keys": "tab"},
                                  effect={"focused": focus_after.display()},
                                  x=x, y=y, confidence=0.6, **common)

        # 8. Generic click: only when the cursor settled *inside a region that
        #    actually changed*. Otherwise abstain — a settled cursor over an
        #    unchanged area is not evidence of a click (abstain over guess).
        if clicked and cursor_pt is not None and _near_changed(cursor_pt, match, before, after):
            control = (after.at(*cursor_pt) or before.at(*cursor_pt)
                       or after.near(*cursor_pt) or before.near(*cursor_pt))
            if control is not None:
                x, y = control.center
                return InferredAction("click", target=control,
                                      x=x, y=y, confidence=0.55, **common)

        return InferredAction("noop", confidence=0.0, **common)


class StateTrajectoryBuilder:
    """Compose proposed transitions + inferred actions into a Trajectory.

    The output is the same :class:`Trajectory` the VLM/scripted paths produce, so
    it serializes to a raw trace and flows through normalize -> induction.
    """

    def __init__(self, cursor: Optional[CursorTrack] = None) -> None:
        self.cursor = cursor or CursorTrack.empty()
        self.proposer = TransitionProposer()
        self.idm = StateDiffIDM(self.cursor)

    def build(self, states: List[ScreenState], *, video_id: str,
              source: Optional[str] = None) -> Trajectory:
        actions: List[VideoAction] = []
        idx = 0
        inferred: List[InferredAction] = []
        for tr in self.proposer.propose(states):
            action = self.idm.infer(tr.before, tr.after)
            if action.action_type == "noop":
                continue
            inferred.append(action)
            actions.append(action.to_video_action(idx, tr.before.ms, tr.after.ms))
            idx += 1
        traj = Trajectory(video_id=video_id, actions=actions, source=source)
        traj.inferred = inferred  # type: ignore[attr-defined]  # rich actions for the graph
        return traj
