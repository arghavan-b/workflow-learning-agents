"""Resolve a workflow ``target`` to a concrete page element.

Video-derived skills carry no stable selector (and even DOM-recorded selectors
like React's ``input#_r_22_`` are volatile), so grounding cannot rely on an
exact-selector shortcut. The strategy is ordered from most to least precise and
returns a confidence score; below :data:`GROUNDING_THRESHOLD` the executor
refuses to act and routes to the repair loop instead of clicking blindly.
"""

from __future__ import annotations

import difflib
import re
from typing import List, Optional

from demo2skill.executor.models import GroundingResult
from demo2skill.executor.page import Element, Page
from demo2skill.workflow.schema import Target


def _norm(value: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip().lower()


def _tokens(value: Optional[str]) -> set:
    return {t for t in re.split(r"\W+", _norm(value)) if len(t) > 2}


def _role_compatible(target: Target, el: Element) -> bool:
    if not target.role:
        return True
    return _norm(target.role) == _norm(el.role)


def _fuzzy(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _norm(a), _norm(b)).ratio()


class Grounder:
    """Ordered, confidence-scored target -> element resolution.

    ``strict`` mode uses only exact identifier tiers (selector/label/aria/
    placeholder/text): the executor acts deliberately and routes anything it
    can't pin down to the repair loop. The repair loop re-grounds with
    ``strict=False`` (semantic token/fuzzy fallback) - the asymmetry is what
    makes self-healing observable instead of silently masked by fuzzy matching.
    """

    def ground(self, target: Target, page: Page, *, strict: bool = False) -> GroundingResult:
        best = GroundingResult(found=False)

        for el in page.elements:
            result = self._score(target, el, strict=strict)
            if result.confidence > best.confidence:
                best = result
        return best

    def _score(self, target: Target, el: Element, *, strict: bool = False) -> GroundingResult:
        h = el.handle

        # 1. exact selector (rarely survives, but cheap and decisive)
        if target.selector and el.selector and _norm(target.selector) == _norm(el.selector):
            return GroundingResult(True, h, 0.97, "selector")

        # 2-4. exact identifier matches
        if target.label and el.label and _norm(target.label) == _norm(el.label):
            return GroundingResult(True, h, 0.92, "label")
        if target.aria_label and el.aria_label and _norm(target.aria_label) == _norm(el.aria_label):
            return GroundingResult(True, h, 0.90, "aria_label")
        if target.placeholder and el.placeholder and _norm(target.placeholder) == _norm(el.placeholder):
            return GroundingResult(True, h, 0.88, "placeholder")

        # 5. exact visible text (click targets: buttons / links)
        if target.text and el.text and _norm(target.text) == _norm(el.text):
            return GroundingResult(True, h, 0.86, "text")

        if strict:
            # Deliberate execution: no exact match means we do not act - the
            # repair loop will re-ground semantically and patch the target.
            return GroundingResult(False)

        # 6. semantic fallback: role-compatible + fuzzy/token overlap across all
        #    of the target's and element's human-readable identifiers. This is
        #    the "semantic_search_then_retry" recovery the repair loop leans on.
        if not _role_compatible(target, el):
            return GroundingResult(False)

        target_terms = [t for t in (target.label, target.text, target.aria_label,
                                     target.placeholder, target.nearby_text) if t]
        if not target_terms:
            # role-only target: weak evidence, but usable if role matches.
            return GroundingResult(True, h, 0.55, "role") if target.role else GroundingResult(False)

        el_terms = el.search_terms()
        token_overlap = 0.0
        fuzzy_best = 0.0
        for tt in target_terms:
            for et in el_terms:
                inter = _tokens(tt) & _tokens(et)
                union = _tokens(tt) | _tokens(et)
                if union:
                    token_overlap = max(token_overlap, len(inter) / len(union))
                fuzzy_best = max(fuzzy_best, _fuzzy(tt, et))

        score = max(token_overlap, fuzzy_best * 0.85)
        # Keep semantic matches below exact-match tiers so they never outrank a
        # true exact hit, but above the act/repair threshold when convincing.
        confidence = round(min(0.8, 0.45 + 0.45 * score), 3)
        if score >= 0.3:
            return GroundingResult(True, h, confidence, "role_fuzzy")
        return GroundingResult(False)


def best_relabel(target: Target, page: Page, exclude_handle: Optional[str] = None) -> Optional[Element]:
    """For repair: find the element the (now-stale) target most likely meant,
    ignoring its broken selector/label, and return it so a patch can adopt its
    current label. Returns ``None`` if nothing is convincing."""

    grounder = Grounder()
    # Re-ground with selector stripped so a dead selector can't dominate.
    relaxed = target.model_copy(update={"selector": None})
    candidates: List[tuple] = []
    for el in page.elements:
        if el.handle == exclude_handle:
            continue
        res = grounder._score(relaxed, el)
        if res.found:
            candidates.append((res.confidence, el))
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0], reverse=True)
    return candidates[0][1]
