"""Concrete ``ScreenParserClient`` backends for the pixels->ScreenState front.

These wrap a real vision model behind the one-method ``complete`` contract that
:class:`~demo2skill.video.statediff.parser.vlm.VLMScreenParser` expects. Two
references are provided, both with lazy imports so the base package never
depends on a model runtime:

* :class:`TransformersScreenVLMClient` - loads a local ScreenVLM / Qwen2-VL-style
  checkpoint via HuggingFace ``transformers``. Install with the ``screenvlm``
  extra (``uv sync --extra screenvlm``). This is the on-device path the paper
  motivates (a compact 316M parser).
* :class:`AnthropicVisionClient` - sends frames to a Claude vision model, reusing
  the existing ``llm`` extra. Zero extra install if you already use the LLM
  induction path; handy for trying the pipeline before standing up a local model.

``default_screen_parser_client`` picks one from the environment, mirroring
:func:`demo2skill.induction.llm.default_client`.
"""

from __future__ import annotations

import base64
import io
import os
from typing import List, Optional

from demo2skill.video.statediff.parser.vlm import ScreenParserClient


class TransformersScreenVLMClient:
    """``ScreenParserClient`` backed by a HuggingFace image-text-to-text model.

    Targets ScreenVLM's checkpoint or any Qwen2-VL-compatible vision LLM. The
    model is loaded once in ``__init__``; each ``complete`` runs one generation
    over a single frame and returns the raw decoded text (the dense-parse JSON
    the prompt asks for).
    """

    def __init__(
        self,
        model_id: str = "screenvlm/screenvlm-316m",
        *,
        device_map: str = "auto",
        max_new_tokens: int = 1024,
        dtype: Optional[str] = None,
    ) -> None:
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:  # pragma: no cover - exercised only with extra
            raise SystemExit(
                "The 'transformers' + 'torch' packages are required for the "
                "TransformersScreenVLMClient.\n"
                "Install them with:  uv sync --extra screenvlm"
            ) from exc

        self._torch = torch
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(model_id)
        kwargs = {"device_map": device_map}
        if dtype is not None:
            kwargs["torch_dtype"] = getattr(torch, dtype, None) or dtype
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)

    def complete(self, *, system: str, prompt: str, images: List[bytes]) -> str:
        from PIL import Image

        pil_images = [Image.open(io.BytesIO(b)).convert("RGB") for b in images]
        content = []
        content.extend({"type": "image"} for _ in pil_images)
        content.append({"type": "text", "text": f"{system}\n\n{prompt}"})
        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            images=pil_images or None,
            return_tensors="pt",
        ).to(self.model.device)

        with self._torch.no_grad():
            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        # Drop the prompt tokens; decode only the newly generated continuation.
        new_tokens = generated[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]


class AnthropicVisionClient:
    """``ScreenParserClient`` backed by a Claude vision model (reuses ``llm`` extra)."""

    def __init__(
        self,
        model: str = "claude-opus-4-8",
        *,
        api_key: Optional[str] = None,
        max_tokens: int = 2048,
        media_type: str = "image/png",
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - exercised only with extra
            raise SystemExit(
                "The 'anthropic' package is required for the AnthropicVisionClient.\n"
                "Install it with:  uv sync --extra llm"
            ) from exc
        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens
        self.media_type = media_type

    def complete(self, *, system: str, prompt: str, images: List[bytes]) -> str:
        content: list = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": self.media_type,
                    "data": base64.b64encode(b).decode("ascii"),
                },
            }
            for b in images
        ]
        content.append({"type": "text", "text": prompt})
        message = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": content}],
        )
        return "".join(block.text for block in message.content if block.type == "text")


def default_screen_parser_client() -> Optional[ScreenParserClient]:
    """Pick a client from the environment, or ``None`` (no model configured).

    * ``SCREENVLM_MODEL`` set -> local transformers checkpoint of that id;
    * else ``ANTHROPIC_API_KEY`` set -> Claude vision;
    * else ``None`` - the caller should supply states another way (e.g. the
      scripted parser).
    """

    model_id = os.environ.get("SCREENVLM_MODEL")
    if model_id:
        try:
            return TransformersScreenVLMClient(model_id)
        except SystemExit:
            return None
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return AnthropicVisionClient()
        except SystemExit:
            return None
    return None
