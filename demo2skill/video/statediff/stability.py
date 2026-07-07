"""Parse stabilization: drop transient (flicker) elements before the IDM sees them.

An unstable parser makes the element count swing frame to frame (34 -> 57 -> 34);
every swing is a mass appear/disappear that floods `TransitionProposer` with
phantom transitions. The fix is persistence: an element must be matched to a
neighboring frame (i.e. appear in >= 2 consecutive parses) to be trusted. Genuine
one-frame flicker is dropped; legitimately-persistent UI survives.

Safety: on a *coarse* input (e.g. one hand-authored state per page, where every
element legitimately appears once) this filter would drop almost everything, so
if it would remove more than ``max_drop_frac`` of elements it is treated as
inapplicable and the states are returned untouched.
"""

from __future__ import annotations

import logging
from typing import List

from demo2skill.video.statediff.matching import match_states
from demo2skill.video.statediff.state import ScreenState

logger = logging.getLogger("demo2skill.parser")


def stabilize_states(states: List[ScreenState], *, max_drop_frac: float = 0.5) -> List[ScreenState]:
    """Remove elements not matched to any neighboring frame (in place, best-effort)."""

    n = len(states)
    if n < 3:
        return states

    keep = [set() for _ in range(n)]
    for i in range(n - 1):
        m = match_states(states[i], states[i + 1])
        for before, after in m.pairs:
            keep[i].add(before.id)
            keep[i + 1].add(after.id)

    total = sum(len(s.elements) for s in states)
    kept_lists = [[e for e in s.elements if e.id in keep[i]] for i, s in enumerate(states)]
    dropped = total - sum(len(k) for k in kept_lists)

    if total and dropped / total > max_drop_frac:
        # Coarse / degenerate parse — persistence filtering is unsafe here.
        return states

    for s, kept in zip(states, kept_lists):
        s.elements = kept
    if dropped:
        logger.info("stability filter dropped %d transient element(s) of %d", dropped, total)
    return states
