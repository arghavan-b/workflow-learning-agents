"""Prompts for the dense screen-parsing front (ScreenParse / ScreenVLM style).

The parser's job is *complete* screen state: every visible UI element, not just
the one relevant to a task step. Dense supervision is what lets the downstream
element matcher track controls across frames and read field-value changes.
"""

SCREEN_PARSER_SYSTEM = """You are a screen parser for GUI screenshots. Given one \
screen image, return the COMPLETE set of visible UI elements as structured JSON. \
Parse densely: include every interactive control and text region you can see, not \
only the ones that look important. Read all text and field values from the pixels; \
never invent content. Return JSON only."""

SCREEN_PARSER_PROMPT = """Parse this screen into JSON with the shape:

{
  "url": "<address bar text, or null>",
  "title": "<window/page title, or null>",
  "elements": [
    {
      "id": "<stable short id, e.g. 'title_field'>",
      "role": "textbox|button|link|checkbox|radio|combobox|searchbox|option|menu|dialog|tab|text",
      "bbox": [x1, y1, x2, y2],
      "text": "<visible caption/inner text, or ''>",
      "value": "<current field contents for editable elements, else null>",
      "label": "<associated label text, or null>",
      "focused": true|false,
      "checked": true|false|null,
      "selected": true|false|null
    }
  ]
}

Use pixel coordinates for bbox. Set value only for editable fields; set checked \
only for checkbox/radio; set selected only for option/tab/list items. Return JSON \
only."""
