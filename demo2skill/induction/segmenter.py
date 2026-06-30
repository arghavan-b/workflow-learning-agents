"""Clean and segment a normalized trace into workflow steps (Module 3).

Real demonstrations are noisy: every keystroke is its own ``fill_field`` event,
humans log in, browsers write hidden support flags, and people explore before
doing the task. The deterministic baseline here removes that noise without an
LLM:

* hidden environment writes (``webauthn-support`` ...) are dropped
* login interactions become a ``user_logged_in`` precondition, not steps
* per-keystroke ``fill_field`` runs collapse to the final committed value
* focus clicks that just precede typing into the same field are dropped
* the navigation chain before the first form interaction collapses to a single
  ``navigate`` to the form page (its net effect is "be on the form")

The result is grouped into :class:`Segment` objects. An optional LLM segmenter
can replace this stage; see :mod:`demo2skill.induction.workflow_generator`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

# Hidden inputs frameworks write on load; never part of a human's intent.
ENV_FIELD_PATTERN = re.compile(r"(webauthn|javascript)[-_]", re.IGNORECASE)
# Pages / fields that indicate authentication, captured as a precondition.
LOGIN_URL_PATTERN = re.compile(r"/(login|session|signin|sign_in|sso)\b", re.IGNORECASE)
LOGIN_LABELS = {"username or email address", "password", "username", "email", "login"}

NAVIGATIONAL = {"navigate", "click"}
CONTENT_ACTIONS = {"fill_field", "upload_file", "set_value"}


@dataclass
class CleanEvent:
    action: str
    target: Dict[str, Any] = field(default_factory=dict)
    value: Optional[str] = None
    url: Optional[str] = None
    page_title: Optional[str] = None
    source_event_ids: List[str] = field(default_factory=list)
    key: Optional[str] = None  # stable field identity used for collapsing


@dataclass
class Segment:
    segment_id: str
    name: str
    intent: str
    events: List[CleanEvent]
    essential: bool = True

    @property
    def event_ids(self) -> List[str]:
        ids: List[str] = []
        for ev in self.events:
            ids.extend(ev.source_event_ids)
        return ids

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "name": self.name,
            "intent": self.intent,
            "events": self.event_ids,
            "essential": self.essential,
        }


def _target(event: Mapping[str, Any]) -> Dict[str, Any]:
    target = event.get("target")
    return dict(target) if isinstance(target, Mapping) else {}


def _field_key(target: Mapping[str, Any]) -> str:
    for key in ("selector", "name", "id", "label", "text"):
        value = target.get(key)
        if value:
            return str(value)
    return "unknown"


def _is_env_field(target: Mapping[str, Any]) -> bool:
    blob = " ".join(
        str(target.get(k) or "") for k in ("name", "id", "label", "selector")
    )
    return bool(ENV_FIELD_PATTERN.search(blob))


def _is_login(event: Mapping[str, Any], target: Mapping[str, Any]) -> bool:
    url = str(event.get("url") or "")
    if LOGIN_URL_PATTERN.search(url):
        return True
    label = str(target.get("label") or target.get("name") or target.get("id") or "").lower()
    return label in LOGIN_LABELS


@dataclass
class CleanResult:
    events: List[CleanEvent]
    preconditions: List[str]


def clean_events(events: List[Mapping[str, Any]]) -> CleanResult:
    """Collapse keystroke noise and strip accidental / environmental actions."""

    cleaned: List[CleanEvent] = []
    preconditions: List[str] = []

    for event in events:
        action = str(event.get("semantic_action") or "unknown")
        target = _target(event)
        url = event.get("url")

        if action in ("fill_field", "set_value") and _is_env_field(target):
            continue
        if _is_login(event, target):
            if "user_logged_in" not in preconditions:
                preconditions.append("user_logged_in")
            continue

        if action == "navigate":
            nav_url = target.get("url") or url
            if not nav_url or nav_url == "about:blank":
                continue
            if cleaned and cleaned[-1].action == "navigate" and cleaned[-1].url == nav_url:
                cleaned[-1].source_event_ids.append(event.get("event_id"))
                continue
            cleaned.append(
                CleanEvent("navigate", {}, None, nav_url, event.get("page_title"),
                           [event.get("event_id")])
            )
            continue

        if action in ("fill_field", "set_value"):
            key = _field_key(target)
            value = event.get("value")
            if cleaned and cleaned[-1].action == "fill_field" and cleaned[-1].key == key:
                if value not in (None, ""):
                    cleaned[-1].value = value
                    cleaned[-1].target = target or cleaned[-1].target
                cleaned[-1].source_event_ids.append(event.get("event_id"))
                continue
            cleaned.append(
                CleanEvent("fill_field", target, value, url, event.get("page_title"),
                           [event.get("event_id")], key=key)
            )
            continue

        # click / upload_file / select_option / set_checked / extract_text ...
        normalized = "upload_file" if action == "upload_file" else action
        cleaned.append(
            CleanEvent(normalized, target, event.get("value"), url, event.get("page_title"),
                       [event.get("event_id")], key=_field_key(target))
        )

    cleaned = _drop_focus_clicks_and_empty_fills(cleaned)
    cleaned = _collapse_navigation_prefix(cleaned)
    return CleanResult(cleaned, preconditions)


def _drop_focus_clicks_and_empty_fills(events: List[CleanEvent]) -> List[CleanEvent]:
    result: List[CleanEvent] = []
    for i, event in enumerate(events):
        nxt = events[i + 1] if i + 1 < len(events) else None
        if (
            event.action == "click"
            and nxt is not None
            and nxt.action == "fill_field"
            and nxt.key == event.key
        ):
            nxt.source_event_ids = event.source_event_ids + nxt.source_event_ids
            continue
        if event.action == "fill_field" and event.value in (None, ""):
            continue
        result.append(event)
    return result


def _collapse_navigation_prefix(events: List[CleanEvent]) -> List[CleanEvent]:
    """Collapse the leading navigate/click chain into one navigate to the form."""

    first_content = next(
        (i for i, e in enumerate(events) if e.action in CONTENT_ACTIONS), None
    )
    if first_content in (None, 0):
        return events
    prefix = events[:first_content]
    if not all(e.action in NAVIGATIONAL for e in prefix):
        return events

    dest_url = events[first_content].url
    if not dest_url:
        navs = [e.url for e in prefix if e.action == "navigate" and e.url]
        dest_url = navs[-1] if navs else None
    if not dest_url:
        return events

    source_ids: List[str] = []
    for event in prefix:
        source_ids.extend(event.source_event_ids)
    nav = CleanEvent("navigate", {}, None, dest_url,
                     prefix[-1].page_title, source_ids)
    return [nav] + events[first_content:]


# -- segmentation ----------------------------------------------------------


def _slug(text: str, *, max_words: int = 4) -> str:
    words = re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
    return "_".join(words[:max_words]) or "step"


def _segment_name(events: List[CleanEvent]) -> str:
    first = events[0]
    if first.action == "navigate":
        path = re.sub(r"^https?://[^/]+", "", first.url or "").strip("/")
        return f"navigate_to_{_slug(path) or 'page'}"
    if all(e.action == "fill_field" for e in events):
        fields = "_".join(_slug(e.target.get("label") or e.target.get("text") or "", max_words=2)
                          for e in events)
        return f"fill_{_slug(fields, max_words=6)}"
    if first.action == "click":
        return f"click_{_slug(first.target.get('text') or first.target.get('label') or '')}"
    return _slug(first.action)


def _group_key(event: CleanEvent) -> str:
    if event.action == "navigate":
        return "nav"
    if event.action == "fill_field":
        return f"fill::{event.url}"
    return f"{event.action}::{event.url}"


def segment_events(events: List[CleanEvent]) -> List[Segment]:
    """Group contiguous cleaned events that share a page and action kind."""

    segments: List[Segment] = []
    current: List[CleanEvent] = []
    current_key: Optional[str] = None

    def flush() -> None:
        if not current:
            return
        sid = f"seg_{len(segments) + 1:03d}"
        segments.append(
            Segment(sid, _segment_name(current), _segment_intent(current), list(current))
        )

    for event in events:
        key = _group_key(event)
        # navigate is always its own boundary.
        if event.action == "navigate" or key != current_key:
            flush()
            current = [event]
            current_key = key
            if event.action == "navigate":
                flush()
                current = []
                current_key = None
        else:
            current.append(event)
    flush()
    return segments


def _segment_intent(events: List[CleanEvent]) -> str:
    first = events[0]
    if first.action == "navigate":
        return f"Open {first.url}"
    if all(e.action == "fill_field" for e in events):
        labels = [e.target.get("label") or e.target.get("text") or "field" for e in events]
        return "Fill " + ", ".join(labels)
    if first.action == "click":
        return f"Click {first.target.get('text') or first.target.get('label') or 'element'}"
    return f"Perform {first.action}"
