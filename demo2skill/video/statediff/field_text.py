"""Classify field text by *when* it appears, not what it looks like (Option 1).

Placeholder text and typed values are indistinguishable in a single frame but
trivially separable across frames:

* a **placeholder** sits in the field at rest and *vanishes* the moment the field
  is focused / the first character is typed;
* a **typed value** starts absent and *grows/persists* through the run.

This is the state-diff philosophy applied to the text inside an element. Using
the element chains that `matching.py` already produces, we follow each editable
field across states, read how its text evolves (the per-frame OCR value), and:

* tag the pre-interaction text that disappears as ``placeholder_text`` (and clear
  it from ``value`` so the IDM never treats it as typed content);
* keep the growing/persisting text as ``value``.

No new model — just `match_states` over the parsed states. When the run's start
isn't captured (video begins mid-task), the disappearance signal is absent and
the field is left untouched (abstain, don't guess).
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import List, Optional, Tuple

from demo2skill.video.statediff.matching import match_states
from demo2skill.video.statediff.state import ScreenState, UIElement

logger = logging.getLogger("demo2skill.parser")


def _norm(s) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _similar(a: str, b: str) -> bool:
    """Fuzzy equality tolerant of OCR noise (a placeholder read slightly wrong)."""

    a, b = _norm(a), _norm(b)
    if not a or not b:
        return a == b
    d = _levenshtein(a, b)
    return d <= 2 or d <= 0.2 * max(len(a), len(b))


def vote_value(readings: List[str]) -> Optional[str]:
    """Fuse many noisy OCR reads of the same growing string into one (Option 4).

    Whole-string voting (no per-character alignment, so it can't invent doubled
    characters). Restrict to the near-complete frames (>= 80% of the longest
    read) so growth prefixes don't dominate; take the most frequent read. On a
    tie (all reads unique under noise) pick the **medoid** — the read with the
    smallest total edit distance to the pool — not merely the longest, so the
    noisiest single read can't win.
    """

    texts = [r.strip() for r in readings if r and r.strip()]
    if not texts:
        return None
    max_len = max(len(t) for t in texts)
    pool = [t for t in texts if len(t) >= max(1, round(max_len * 0.8))] or texts
    counts = Counter(pool)
    best = max(counts.values())
    cands = [t for t, n in counts.items() if n == best]
    if len(cands) == 1:
        return cands[0]
    return min(cands, key=lambda c: sum(_levenshtein(_norm(c), _norm(p)) for p in pool))


def _reading(el: UIElement) -> str:
    """The field's current visible text (OCR value, else element text)."""

    if el.value not in (None, ""):
        return str(el.value)
    return el.text or ""


def chain_editables(states: List[ScreenState]) -> List[List[Tuple[int, UIElement]]]:
    """Follow each editable element across states via `match_states`.

    Returns a list of chains; each chain is ``[(state_index, element), ...]`` in
    time order for one tracked field.
    """

    chains: List[List[Tuple[int, UIElement]]] = []
    active: dict = {}  # element id in the previous state -> its chain

    for i, state in enumerate(states):
        editable_ids = {e.id for e in state.elements if e.editable}
        if i == 0:
            for e in state.elements:
                if e.editable:
                    ch = [(i, e)]
                    chains.append(ch)
                    active[e.id] = ch
            continue

        match = match_states(states[i - 1], state)
        new_active: dict = {}
        matched_after = set()
        for before, after in match.pairs:
            if after.id in editable_ids and before.id in active:
                ch = active[before.id]
                ch.append((i, after))
                new_active[after.id] = ch
                matched_after.add(after.id)
        # editable elements with no match start a fresh chain
        for e in state.elements:
            if e.editable and e.id not in matched_after:
                ch = [(i, e)]
                chains.append(ch)
                new_active[e.id] = ch
        active = new_active

    return chains


def classify_field_text(states: List[ScreenState]) -> List[ScreenState]:
    """Reclassify placeholder vs typed value across a run, in place.

    For each editable field chain, the first non-empty reading is a *placeholder*
    if it later vanishes (a frame reads empty) or is replaced by different,
    non-extending text. Wherever that placeholder text appears it is moved to
    ``placeholder_text`` and cleared from ``value``; growing/typed text is kept.
    """

    for chain in chain_editables(states):
        if len(chain) < 2:
            continue

        seq = [(idx, el, _reading(el)) for idx, el in chain]
        nonempty = [(idx, el, t) for idx, el, t in seq if t.strip()]
        if not nonempty:
            continue

        first_idx, _first_el, first_text = nonempty[0]
        first_n = _norm(first_text)
        later = [t for idx, el, t in seq if idx > first_idx]
        vanished = any(not t.strip() for t in later)
        # Replaced by *substantially longer* different text (a placeholder giving
        # way to typed content). The length gate keeps same-length OCR noise
        # ("Bug in login" -> "Bug 1n login") from looking like a replacement.
        replaced = any(t.strip() and not _norm(t).startswith(first_n)
                       and len(_norm(t)) >= 1.5 * len(first_n)
                       for t in later)
        if not (vanished or replaced):
            continue  # text persisted -> a real pre-existing value; leave it

        placeholder = first_text
        for _idx, el, t in seq:
            # Fuzzy match: catch noisy OCR variants of the placeholder too, so
            # they don't survive into the value and contaminate the vote.
            if t.strip() and _similar(t, placeholder):
                el.placeholder_text = placeholder
                el.value = None
                el.text = ""       # Bug 1: else _value() falls back to the text
        logger.info("placeholder classified: %r (field chain of %d frames)",
                    placeholder, len(chain))

    _vote_field_values(states)
    return states


def _vote_field_values(states: List[ScreenState]) -> None:
    """Option 4: replace each field's *typed* value with a temporal vote across
    the run. Only frames that still carry a value after placeholder removal are
    voted and rewritten, so pre-typing frames stay empty and the empty->typed
    transition (the `type` event) is preserved."""

    for chain in chain_editables(states):
        if len(chain) < 2:
            continue
        typed = [(idx, el) for idx, el in chain if el.value not in (None, "")]
        if len(typed) < 2:
            continue
        voted = vote_value([el.value for _idx, el in typed])
        if not voted:
            continue
        changed = any(str(el.value) != voted for _idx, el in typed)
        for _idx, el in typed:      # only the typed frames, never pre-typing ones
            el.value = voted
        if changed:
            logger.info("voted field value across %d frames -> %r", len(typed), voted)
