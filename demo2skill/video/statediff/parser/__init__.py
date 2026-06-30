"""Pixels->ScreenState front for the state-diff engine.

This is the pluggable parser slot statediff assumes but does not implement
inline: it converts frames into the parsed :class:`ScreenState`s the
inverse-dynamics module reasons over. ``VLMScreenParser`` is the ScreenVLM /
ScreenParse-style model backend; ``ScriptedScreenParser`` replays pre-parsed
states for tests. ``parse_frames`` drives either over a frame stream.
"""

from demo2skill.video.statediff.parser.base import (
    ScreenParser,
    ScriptedScreenParser,
    build_element,
    build_state,
    load_states,
    parse_frames,
)
from demo2skill.video.statediff.parser.vlm import ScreenParserClient, VLMScreenParser
from demo2skill.video.statediff.parser.clients import (
    AnthropicVisionClient,
    TransformersScreenVLMClient,
    default_screen_parser_client,
)

__all__ = [
    "ScreenParser",
    "ScriptedScreenParser",
    "VLMScreenParser",
    "ScreenParserClient",
    "build_element",
    "build_state",
    "load_states",
    "parse_frames",
    # concrete model backends (lazy deps)
    "TransformersScreenVLMClient",
    "AnthropicVisionClient",
    "default_screen_parser_client",
]
