"""VLM-backed IDM stages: the seam for a real grounding/recognition model.

A ``VLMClient`` is the minimal contract (one multimodal completion call). Plug
in Qwen-VL, Claude, or any vision LLM. The detector and recognizer here build
the prompts (see :mod:`demo2skill.video.video2action.prompts`), pass the relevant frames, and
parse the JSON back into :class:`ActionInterval` / :class:`VideoAction`. No model
is bundled - this module is import-safe without any VLM installed.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from demo2skill.video.video2action import prompts
from demo2skill.video.video2action.frames import Frame, Frames
from demo2skill.video.video2action.idm import ActionInterval
from demo2skill.video.schema import VideoAction


@runtime_checkable
class VLMClient(Protocol):
    """Minimal multimodal-completion contract."""

    def complete(self, *, system: str, prompt: str, images: List[bytes]) -> str:
        """Return the model's text response given a prompt and frame images."""
        ...


def _parse_json(text: str) -> Any:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    return json.loads(text)


def _frame_index_table(frames: Frames) -> str:
    return "\n".join(f"{f.index}: {f.ms}" for f in frames.frames) or "(no frames)"


def _images(window: List[Frame]) -> List[bytes]:
    return [b for b in (f.bytes() for f in window) if b is not None]


class VLMTemporalDetector:
    def __init__(self, client: VLMClient) -> None:
        self.client = client

    def detect(self, frames: Frames) -> List[ActionInterval]:
        text = self.client.complete(
            system=prompts.TEMPORAL_DETECTOR_SYSTEM,
            prompt=prompts.TEMPORAL_DETECTOR_PROMPT.format(
                frame_index=_frame_index_table(frames)
            ),
            images=_images(frames.frames),
        )
        out: List[ActionInterval] = []
        for rec in _parse_json(text):
            out.append(ActionInterval(
                action_type=str(rec.get("action", "click")),
                start_ms=int(rec.get("start_ms", 0)),
                end_ms=int(rec.get("end_ms", rec.get("start_ms", 0))),
                confidence=float(rec.get("confidence", 1.0)),
            ))
        return out


class VLMContentRecognizer:
    def __init__(self, client: VLMClient, pad_ms: int = 600) -> None:
        self.client = client
        self.pad_ms = pad_ms

    def recognize(self, index: int, interval: ActionInterval, frames: Frames) -> VideoAction:
        window = frames.window(interval.start_ms - self.pad_ms, interval.end_ms + self.pad_ms)
        text = self.client.complete(
            system=prompts.CONTENT_RECOGNIZER_SYSTEM,
            prompt=prompts.CONTENT_RECOGNIZER_PROMPT.format(action_type=interval.action_type),
            images=_images(window),
        )
        rec: Dict[str, Any] = _parse_json(text)
        return VideoAction(
            index=index,
            action_type=interval.action_type,
            start_ms=interval.start_ms,
            end_ms=interval.end_ms,
            x=rec.get("x"), y=rec.get("y"),
            text=rec.get("text"), keys=rec.get("keys"), url=rec.get("url"),
            target_text=rec.get("target_text"),
            target_label=rec.get("target_label"),
            target_role=rec.get("target_role"),
            page_title=rec.get("page_title"),
            confidence=float(rec.get("confidence", interval.confidence)),
        )
