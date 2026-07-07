"""The concrete ScreenVLM clients stay import-safe without their model runtimes.

Mirrors how the Anthropic induction client is wired: importing the module must
never require torch / transformers / anthropic, the env-driven default returns
``None`` when nothing is configured, and asking for a backend whose dependency
is absent raises a helpful install message rather than a bare ImportError.
"""

from __future__ import annotations

import importlib
import unittest

from demo2skill.video.statediff.parser import clients
from demo2skill.video.statediff.parser.vlm import ScreenParserClient


def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


class ImportSafetyTest(unittest.TestCase):
    def test_module_imports_without_model_runtimes(self):
        # The mere import above already proves base-package safety; assert the
        # public surface is present.
        self.assertTrue(hasattr(clients, "TransformersScreenVLMClient"))
        self.assertTrue(hasattr(clients, "AnthropicVisionClient"))
        self.assertTrue(hasattr(clients, "default_screen_parser_client"))


class DefaultClientTest(unittest.TestCase):
    def test_returns_none_when_nothing_configured(self):
        import os

        saved = {k: os.environ.pop(k, None) for k in ("SCREENVLM_MODEL", "ANTHROPIC_API_KEY")}
        try:
            self.assertIsNone(clients.default_screen_parser_client())
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v


class MissingDependencyTest(unittest.TestCase):
    def test_transformers_client_raises_helpful_message_when_absent(self):
        if _installed("torch") and _installed("transformers"):
            self.skipTest("transformers/torch installed; absence path not exercised here")
        with self.assertRaises(SystemExit) as ctx:
            clients.TransformersScreenVLMClient()
        self.assertIn("screenvlm", str(ctx.exception))

    def test_anthropic_vision_client_raises_helpful_message_when_absent(self):
        if _installed("anthropic"):
            self.skipTest("anthropic installed; absence path not exercised here")
        with self.assertRaises(SystemExit) as ctx:
            clients.AnthropicVisionClient()
        self.assertIn("llm", str(ctx.exception))

    def test_openai_vision_client_raises_helpful_message_when_absent(self):
        if _installed("openai"):
            self.skipTest("openai installed; absence path not exercised here")
        with self.assertRaises(SystemExit) as ctx:
            clients.OpenAIVisionClient()
        self.assertIn("openai", str(ctx.exception))


class FakeScreenVLM:
    """A hand-rolled client to confirm the protocol shape is satisfiable."""

    def complete(self, *, system: str, prompt: str, images) -> str:
        return "{}"


class ProtocolConformanceTest(unittest.TestCase):
    def test_fake_client_satisfies_protocol(self):
        self.assertIsInstance(FakeScreenVLM(), ScreenParserClient)


if __name__ == "__main__":
    unittest.main()
