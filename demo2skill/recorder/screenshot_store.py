"""Screenshot artifact storage for recorder events."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


class ScreenshotStore:
    """Save event screenshots under a trace output directory."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.screenshot_dir = self.output_dir / "screens"
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

    def capture(self, page: Any, event_id: str, *, full_page: bool = True) -> Optional[str]:
        path = self.screenshot_dir / f"{event_id}.png"
        try:
            page.screenshot(path=str(path), full_page=full_page)
        except Exception:
            return None
        return str(path.relative_to(self.output_dir))

