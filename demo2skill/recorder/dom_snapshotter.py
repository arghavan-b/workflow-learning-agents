"""DOM and accessibility snapshot helpers for browser demonstrations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


class DomSnapshotter:
    """Persist per-event page snapshots under a trace output directory."""

    def __init__(self, output_dir: Path, max_dom_chars: int = 500_000) -> None:
        self.output_dir = Path(output_dir)
        self.max_dom_chars = max_dom_chars
        self.dom_dir = self.output_dir / "dom"
        self.ax_dir = self.output_dir / "accessibility"
        self.dom_dir.mkdir(parents=True, exist_ok=True)
        self.ax_dir.mkdir(parents=True, exist_ok=True)

    def capture_dom(self, page: Any, event_id: str) -> Optional[str]:
        """Write the current document HTML and return a trace-relative path."""

        try:
            html = page.evaluate(
                """(maxChars) => {
                    const html = document.documentElement.outerHTML || "";
                    return html.length > maxChars ? html.slice(0, maxChars) : html;
                }""",
                self.max_dom_chars,
            )
        except Exception:
            return None

        path = self.dom_dir / f"{event_id}.html"
        path.write_text(html, encoding="utf-8")
        return str(path.relative_to(self.output_dir))

    def capture_accessibility_tree(self, page: Any, event_id: str) -> Optional[str]:
        """Best-effort accessibility snapshot.

        Some Playwright Python versions expose accessibility snapshots and some
        do not. The recorder treats this as an optional artifact.
        """

        snapshot: Optional[Dict[str, Any]]
        try:
            accessibility = getattr(page, "accessibility", None)
            if accessibility is None:
                return None
            snapshot = accessibility.snapshot()
        except Exception:
            return None

        if snapshot is None:
            return None

        path = self.ax_dir / f"{event_id}.json"
        path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(path.relative_to(self.output_dir))

