"""Pluggable LLM client interface for the induction pipeline.

The deterministic baseline needs no LLM at all. When higher-quality
segmentation / induction is wanted, pass any object implementing
:class:`LLMClient`. An Anthropic-backed implementation (with prompt caching) is
provided behind a lazy import so the base package has no hard ``anthropic``
dependency.
"""

from __future__ import annotations

import os
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Minimal text-completion contract the induction stages rely on."""

    def complete(self, *, system: str, prompt: str) -> str:
        """Return the model's text response for ``prompt`` under ``system``."""
        ...


class AnthropicClient:
    """``LLMClient`` backed by the Anthropic SDK with prompt caching.

    Install the optional dependency with ``pip install .[llm]``. The system
    prompt is marked as a cache breakpoint so repeated inductions reuse it.
    """

    def __init__(
        self,
        model: str = "claude-opus-4-8",
        *,
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - exercised only with extra
            raise SystemExit(
                "The 'anthropic' package is required for the LLM path.\n"
                "Install it with:  uv sync --extra llm"
            ) from exc
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, *, system: str, prompt: str) -> str:
        message = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in message.content if block.type == "text")


def default_client() -> Optional[LLMClient]:
    """Return an Anthropic client if a key is configured, else ``None`` (baseline)."""

    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return AnthropicClient()
        except SystemExit:
            return None
    return None
