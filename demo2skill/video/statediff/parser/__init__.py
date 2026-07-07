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
    state_to_dict,
    states_payload,
)
from demo2skill.video.statediff.parser.vlm import ScreenParserClient, VLMScreenParser
from demo2skill.video.statediff.parser.clients import (
    AnthropicVisionClient,
    OpenAIVisionClient,
    TransformersScreenVLMClient,
    default_screen_parser_client,
)
from demo2skill.video.statediff.parser.screenvlm import (
    ScreenVLMParser,
    parse_screentag,
)
from demo2skill.video.statediff.parser.ocr import (
    OCR,
    EasyOCRBackend,
    PaddleOCRBackend,
    TesseractOCR,
    make_ocr,
    ocr_fill_values,
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
    "state_to_dict",
    "states_payload",
    # concrete model backends (lazy deps)
    "TransformersScreenVLMClient",
    "AnthropicVisionClient",
    "OpenAIVisionClient",
    "default_screen_parser_client",
    # the real ScreenVLM checkpoint (ScreenTag → ScreenState)
    "ScreenVLMParser",
    "parse_screentag",
    # OCR value-fill (ScreenVLM omits typed field text)
    "OCR",
    "TesseractOCR",
    "EasyOCRBackend",
    "PaddleOCRBackend",
    "make_ocr",
    "ocr_fill_values",
]
