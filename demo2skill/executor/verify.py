"""Check whether a step achieved its intended effect.

Rule-based first (never everything-LLM, which is flaky): the verifier reads the
page the same way a person would confirm their work - the field now holds the
value, the field is non-empty, the page shows the expected text. On a video
substrate these reads are OCR rather than DOM ``.value``, so comparisons are
whitespace/case-normalized rather than byte-exact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from demo2skill.executor.grounding import Grounder
from demo2skill.executor.page import Page
from demo2skill.workflow.schema import Check, Postcondition, WorkflowStep
from demo2skill.workflow.schema import Target


def _norm(value: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip().lower()


@dataclass
class CheckResult:
    ok: bool
    detail: str


class Verifier:
    def __init__(self, grounder: Optional[Grounder] = None) -> None:
        self.grounder = grounder or Grounder()

    def verify_checks(self, checks: List[Check], page: Page) -> CheckResult:
        for check in checks:
            result = self._one(check, page)
            if not result.ok:
                return result
        return CheckResult(True, "all checks passed")

    def verify_postcondition(self, post: Optional[Postcondition], page: Page) -> CheckResult:
        if post is None:
            return CheckResult(True, "no postcondition")
        if post.page_contains and _norm(post.page_contains) not in _norm(page.text()):
            return CheckResult(False, f"page does not contain '{post.page_contains}'")
        if post.url_contains and _norm(post.url_contains) not in _norm(page.url):
            return CheckResult(False, f"url does not contain '{post.url_contains}'")
        return CheckResult(True, "postcondition met")

    # -- individual checks ---------------------------------------------------

    def _one(self, check: Check, page: Page) -> CheckResult:
        if check.page_contains is not None:
            if _norm(check.page_contains) in _norm(page.text()):
                return CheckResult(True, "page_contains satisfied")
            return CheckResult(False, f"page missing '{check.page_contains}'")

        if check.field_filled is not None:
            el = self._locate(label=check.field_filled, page=page)
            if el and _norm(el.value):
                return CheckResult(True, f"'{check.field_filled}' is filled")
            return CheckResult(False, f"'{check.field_filled}' is empty or missing")

        if check.field_equals is not None:
            fe = check.field_equals
            el = self._locate(label=fe.label, selector=fe.selector, page=page)
            if el is None:
                return CheckResult(False, f"field '{fe.label or fe.selector}' not found")
            if _norm(el.value) == _norm(fe.value):
                return CheckResult(True, f"field equals '{fe.value}'")
            return CheckResult(
                False, f"field is '{el.value}', expected '{fe.value}'"
            )
        return CheckResult(False, "empty check")

    def _locate(self, *, page: Page, label: Optional[str] = None, selector: Optional[str] = None):
        target = Target(label=label, selector=selector, role="textbox" if label else None)
        res = self.grounder.ground(target, page)
        return page.get(res.element_id) if res.ok else None
