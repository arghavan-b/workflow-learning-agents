"""A dependency-free DOM stand-in for executing and grounding workflow steps.

Parses an HTML string into a flat list of interactive :class:`Element` objects
carrying the same identifying attributes the workflow ``target`` schema uses
(label, role, text, selector, aria_label, placeholder). This lets the executor
and repair loop run and be unit-tested without launching a browser; a real
Playwright page exposes the same surface (``elements``, ``fill``, ``text``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Dict, List, Optional


INTERACTIVE_TAGS = {"input", "textarea", "select", "button", "a"}
CHECKABLE = {"checkbox", "radio"}


def _norm(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", value).strip()
    return text or None


@dataclass
class Element:
    handle: str  # page-local id we can refer to after grounding
    tag: str
    attrs: Dict[str, str] = field(default_factory=dict)
    text: Optional[str] = None
    label: Optional[str] = None
    nearby_text: Optional[str] = None
    value: Optional[str] = None

    @property
    def type(self) -> Optional[str]:
        return _norm(self.attrs.get("type"))

    @property
    def role(self) -> Optional[str]:
        explicit = _norm(self.attrs.get("role"))
        if explicit:
            return explicit
        t = self.type or ""
        if self.tag == "button" or t in {"button", "submit"}:
            return "button"
        if self.tag == "a":
            return "link"
        if self.tag == "textarea":
            return "textbox"
        if self.tag == "select":
            return "combobox"
        if self.tag == "input":
            if t in CHECKABLE:
                return t
            return "textbox"
        return None

    @property
    def selector(self) -> Optional[str]:
        if self.attrs.get("id"):
            return f"{self.tag}#{self.attrs['id']}"
        if self.attrs.get("name"):
            return f"{self.tag}[name='{self.attrs['name']}']"
        return None

    @property
    def aria_label(self) -> Optional[str]:
        return _norm(self.attrs.get("aria-label"))

    @property
    def placeholder(self) -> Optional[str]:
        return _norm(self.attrs.get("placeholder"))

    @property
    def editable(self) -> bool:
        return self.role == "textbox"

    def search_terms(self) -> List[str]:
        """All human-readable identifiers, used for fuzzy grounding."""

        terms = [self.label, self.text, self.aria_label, self.placeholder, self.nearby_text]
        return [t for t in terms if t]


class _Parser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.elements: List[Element] = []
        self._labels_for: Dict[str, str] = {}  # id -> label text
        self._pending_label: Optional[List] = None  # [for_id, text_parts]
        self._open_text: Optional[List] = None  # capture inner text of buttons/links/a
        self._text_chunks: List[str] = []  # rolling page text
        self._counter = 0

    # -- labels -----------------------------------------------------------
    def handle_starttag(self, tag, attrs):
        a = {k: (v or "") for k, v in attrs}
        if tag == "label":
            self._pending_label = [a.get("for"), []]
            return
        if tag in INTERACTIVE_TAGS:
            self._counter += 1
            el = Element(
                handle=f"el_{self._counter:03d}",
                tag=tag,
                attrs=a,
                value=_norm(a.get("value")),
            )
            self.elements.append(el)
            if tag in {"button", "a"}:
                self._open_text = []  # collect inner text into this element

    def handle_endtag(self, tag):
        if tag == "label" and self._pending_label is not None:
            for_id, parts = self._pending_label
            text = _norm(" ".join(parts))
            if for_id and text:
                self._labels_for[for_id] = text
            self._pending_label = None
        if tag in {"button", "a"} and self._open_text is not None:
            text = _norm(" ".join(self._open_text))
            if self.elements:
                self.elements[-1].text = text
            self._open_text = None

    def handle_data(self, data):
        chunk = _norm(data)
        if not chunk:
            return
        self._text_chunks.append(chunk)
        if self._pending_label is not None:
            self._pending_label[1].append(data)
        if self._open_text is not None:
            self._open_text.append(data)


class Page:
    """An executable page: grounded elements + mutable field values."""

    def __init__(self, html: str, url: Optional[str] = None) -> None:
        parser = _Parser()
        parser.feed(html)
        self.url = url
        self._page_text = " ".join(parser._text_chunks)
        self.elements: List[Element] = parser.elements
        # Attach <label for=id> text and a simple nearby-text heuristic.
        for idx, el in enumerate(self.elements):
            el_id = el.attrs.get("id")
            if el_id and el_id in parser._labels_for:
                el.label = parser._labels_for[el_id]
            if el.label is None:
                el.label = el.aria_label or el.placeholder
        self._by_handle = {el.handle: el for el in self.elements}

    @classmethod
    def from_file(cls, path, url: Optional[str] = None) -> "Page":
        from pathlib import Path

        return cls(Path(path).read_text(encoding="utf-8"), url=url)

    def get(self, handle: str) -> Element:
        return self._by_handle[handle]

    def fill(self, handle: str, value: str) -> None:
        self.get(handle).value = value

    def click(self, handle: str) -> None:
        # No-op for the v0 page model; click effects are out of scope here.
        _ = self.get(handle)

    def text(self) -> str:
        """Visible page text plus current field values (for page_contains)."""

        values = " ".join(el.value for el in self.elements if el.value)
        return f"{self._page_text} {values}".strip()
