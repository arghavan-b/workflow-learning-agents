"""ScreenVLM-backed screen parser: the model seam for pixels->ScreenState.

A ``ScreenParserClient`` is the minimal contract - one multimodal completion per
frame. Plug in ScreenVLM, Qwen-VL, Claude, or any vision LLM. The parser builds
the dense-parse prompt (see :mod:`.prompts`), passes the frame image, and parses
the JSON back into a :class:`ScreenState`. No model is bundled, so importing this
module never requires a VLM.

The client contract is intentionally identical to the IDM's ``VLMClient`` (in
:mod:`demo2skill.video.video2action.backends.vlm`), so a single vision-model
wrapper can serve both the parsing front and the action recognizer.
"""

from __future__ import annotations

import json
import re
from typing import Any, List, Optional, Protocol, runtime_checkable

from demo2skill.video.statediff.parser.base import build_state
from demo2skill.video.statediff.parser.prompts import (
    SCREEN_PARSER_PROMPT,
    SCREEN_PARSER_SYSTEM,
)
from demo2skill.video.statediff.state import ScreenState


@runtime_checkable
class ScreenParserClient(Protocol):
    """Minimal multimodal-completion contract (one screen image in, JSON out)."""

    def complete(self, *, system: str, prompt: str, images: List[bytes]) -> str:
        ...


class VLMScreenParser:
    """Parse a frame into a dense :class:`ScreenState` via a vision LLM."""

    def __init__(self, client: ScreenParserClient) -> None:
        self.client = client

    def parse(self, image: Optional[bytes], *, index: int, ms: int) -> ScreenState:
        text = self.client.complete(
            system=SCREEN_PARSER_SYSTEM,
            prompt=SCREEN_PARSER_PROMPT,
            images=[image] if image is not None else [],
        )
        data = _parse_json(text)
        return build_state(data, index=index, ms=ms)


def _parse_json(text: str) -> Any:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    return json.loads(text)
