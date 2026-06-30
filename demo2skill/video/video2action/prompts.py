"""Prompts for the VLM-backed inverse-dynamics stages (VIDEO2ACTION).

Kept separate so they can be tuned without touching control flow, mirroring
:mod:`demo2skill.induction.prompts`.
"""

TEMPORAL_DETECTOR_SYSTEM = """You are a GUI action detector for screen-recorded \
tutorials. Given an ordered, timestamped sequence of screenshots, find every \
discrete user action (click, type, key, scroll, drag, navigate). For each, give \
precise start/end timestamps (ms) and the single best action type. Ignore camera \
movement, idle time, cursor drift, and narration-only segments. Return JSON only."""

TEMPORAL_DETECTOR_PROMPT = """Frames (index: ms):
{frame_index}

Return a JSON list, temporally ordered:
[
  {{"action": "click|type|key|scroll|drag|navigate",
    "start_ms": <int>, "end_ms": <int>, "confidence": <0..1>}}
]"""

CONTENT_RECOGNIZER_SYSTEM = """You extract the structured content of a single GUI \
action from the screenshots spanning it. Report what changed between the frames \
before and after the action. For a click: the click point and the caption/label \
of the control clicked. For typing: the full final text and the field's label. \
Read text from the pixels; never invent it. Return JSON only."""

CONTENT_RECOGNIZER_PROMPT = """Action type: {action_type}
Frames before -> after the action are provided in order.

Return one JSON object:
{{
  "x": <int|null>, "y": <int|null>,
  "text": <typed text|null>,
  "keys": <"ctrl+s"|null>,
  "url": <url if navigation|null>,
  "target_text": <button/link caption|null>,
  "target_label": <field label|null>,
  "target_role": <"textbox"|"button"|"link"|"combobox"|null>,
  "page_title": <window/tab title|null>,
  "confidence": <0..1>
}}"""
