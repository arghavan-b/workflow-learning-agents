"""The real ScreenVLM checkpoint as a `ScreenParser` (ScreenTag → ScreenState).

`docling-project/ScreenVLM` (paper: arXiv:2602.14276) is a compact 316M Idefics3
model that, given a screenshot and a fixed instruction, emits a **ScreenTag**
markup — not JSON — listing every visible UI element with a semantic tag, four
`<loc_N>` tokens (normalized to `[0, 500]`), and its text:

    <screentag>
    <button><loc_10><loc_20><loc_50><loc_35>Submit</button>
    <link><loc_100><loc_200><loc_180><loc_210>Learn more</link>
    ...
    </screentag>

Because its output contract and prompt differ from a general instruct-VLM, it
gets its own `ScreenParser` rather than going through the JSON
`ScreenParserClient` path. `parse_screentag` (a pure function) converts the
markup into `UIElement`s; `ScreenVLMParser` wraps the model. Transformers/torch
are imported lazily, so this module stays import-safe without them.
"""

from __future__ import annotations

import io
import logging
import re
import time
from pathlib import Path
from typing import List, Optional

from demo2skill.video.statediff.parser.ocr import OCR, ocr_fill_values
from demo2skill.video.statediff.state import ScreenState, UIElement

logger = logging.getLogger("demo2skill.parser")

NORM_SIZE = 500
DEFAULT_MODEL = "docling-project/ScreenVLM"
DEFAULT_PROMPT = "Generate the screen representation for this UI:"

# ScreenVLM's flat element pattern (mirrors the model card's reference parser):
# a semantic tag, four location tokens, then the element's text up to the next tag.
_SCREENTAG_RE = re.compile(
    r"<(?P<tag>[a-zA-Z][a-zA-Z0-9_]*)>"
    r"\s*<loc_(?P<l>\d+)><loc_(?P<t>\d+)><loc_(?P<r>\d+)><loc_(?P<b>\d+)>"
    r"(?P<text>[^<]*)"
)

# Map ScreenTag's 55-class vocabulary onto the roles the state-diff IDM reasons
# over. Unmapped tags keep their (lowercased) name — they still participate in
# element matching and churn even if the IDM has no special rule for them.
_ROLE_MAP = {
    "button": "button",
    "link": "link",
    "input": "textbox",
    "text_input": "textbox",
    "textbox": "textbox",
    "text_field": "textbox",
    "input_field": "textbox",
    "textarea": "textbox",
    "text_area": "textbox",
    "search": "searchbox",
    "searchbox": "searchbox",
    "search_box": "searchbox",
    "search_field": "searchbox",
    "checkbox": "checkbox",
    "radio": "radio",
    "radio_button": "radio",
    "dropdown": "combobox",
    "combobox": "combobox",
    "select": "combobox",
    "listbox": "combobox",
    "option": "option",
    "menu_item": "menu",
    "menu": "menu",
    "tab": "tab",
    "dialog": "dialog",
    "modal": "dialog",
}

_EDITABLE = {"textbox", "combobox", "searchbox"}


def map_role(tag: str) -> str:
    return _ROLE_MAP.get(tag.lower(), tag.lower())


def parse_screentag(text: str, width: int, height: int) -> List[UIElement]:
    """Convert ScreenTag markup into `UIElement`s in pixel coordinates.

    `<loc_N>` tokens are on a `[0, NORM_SIZE]` grid; they are rescaled to the
    frame's pixel size. Boxes are normalized to ``(x1, y1, x2, y2)``.
    """

    elements: List[UIElement] = []
    for i, m in enumerate(_SCREENTAG_RE.finditer(text)):
        l, t, r, b = (max(0, min(int(m.group(k)), NORM_SIZE)) for k in ("l", "t", "r", "b"))
        if r < l:
            l, r = r, l
        if b < t:
            t, b = b, t
        x1 = int(round(l / NORM_SIZE * width))
        y1 = int(round(t / NORM_SIZE * height))
        x2 = int(round(r / NORM_SIZE * width))
        y2 = int(round(b / NORM_SIZE * height))
        role = map_role(m.group("tag"))
        content = (m.group("text") or "").strip()
        # ScreenVLM transcribes the visible text of an element; for editable
        # fields that visible text is the field's current content, so surface it
        # as `value` too — that is the signal the IDM uses to detect typing.
        value = content or None if role in _EDITABLE else None
        elements.append(UIElement(
            id=f"el_{i:03d}",
            role=role,
            text=content,
            bbox=(x1, y1, x2, y2),
            value=value,
        ))
    return elements


class ScreenVLMParser:
    """`ScreenParser` backed by the real ScreenVLM checkpoint (Idefics3)."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        *,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        max_new_tokens: int = 6192,
        revision: Optional[str] = None,
        prompt: str = DEFAULT_PROMPT,
        raw_dir: Optional[str] = None,
        max_image_edge: Optional[int] = 1024,
        ocr: Optional[OCR] = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoProcessor
            try:
                from transformers import AutoModelForImageTextToText as _AutoVLM
            except ImportError:  # older transformers
                from transformers import AutoModelForVision2Seq as _AutoVLM
        except ImportError as exc:  # pragma: no cover - exercised only with extra
            raise SystemExit(
                "The 'transformers' + 'torch' packages are required for ScreenVLM.\n"
                "Install them with:  uv sync --extra screenvlm"
            ) from exc

        self._torch = torch
        self.max_new_tokens = max_new_tokens
        self.prompt = prompt
        self.max_image_edge = max_image_edge or None  # 0/None disables downscaling
        self.ocr = ocr                           # fills typed field values ScreenVLM omits
        self.last_raw: Optional[str] = None      # raw ScreenTag of the last frame
        self.raw_dir = Path(raw_dir) if raw_dir else None
        if self.raw_dir:
            self.raw_dir.mkdir(parents=True, exist_ok=True)

        # Device: prefer CUDA, then Apple-Silicon GPU (MPS), else CPU.
        mps_ok = getattr(getattr(torch.backends, "mps", None), "is_available", lambda: False)()
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif mps_ok:
            self.device = "mps"
        else:
            self.device = "cpu"

        if dtype:
            torch_dtype = getattr(torch, dtype)
        elif self.device == "cuda":
            torch_dtype = torch.bfloat16
        elif self.device == "mps":
            torch_dtype = torch.float16    # much faster on Apple GPU
        else:
            torch_dtype = torch.float32

        load_kwargs = {"torch_dtype": torch_dtype}
        if revision:
            load_kwargs["revision"] = revision
        logger.info("loading ScreenVLM '%s' on %s (%s), max_new_tokens=%d ...",
                    model_path, self.device, torch_dtype, max_new_tokens)
        t0 = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(model_path, revision=revision)
        self.model = _AutoVLM.from_pretrained(model_path, **load_kwargs).to(self.device)
        self.model.eval()
        # This checkpoint ships with use_cache=False in its text config, which
        # makes generation recompute the whole sequence every token (O(n^2), ~0.5
        # tok/s). Force the KV cache on — the single biggest speedup here.
        self.model.config.use_cache = True
        text_cfg = getattr(self.model.config, "text_config", None)
        if text_cfg is not None:
            text_cfg.use_cache = True
        if getattr(self.model, "generation_config", None) is not None:
            self.model.generation_config.use_cache = True
        logger.info("ScreenVLM ready in %.1fs on %s", time.perf_counter() - t0, self.device)
        if self.device == "cpu":
            logger.warning("running on CPU — generation is slow; pass --device mps "
                           "on Apple Silicon, or lower --max-new-tokens for a quick test")

    def parse(self, image: Optional[bytes], *, index: int, ms: int) -> ScreenState:
        if image is None:
            return ScreenState(index=index, ms=ms, elements=[])
        from PIL import Image

        pil = Image.open(io.BytesIO(image)).convert("RGB")   # original, full-res
        orig_w, orig_h = pil.size
        # Downscale a *copy* before the model tiles the image: a 2x Retina
        # screenshot otherwise explodes into ~16 tiles and dominates the runtime.
        # The full-res original is kept for OCR crops; boxes are reported in the
        # ORIGINAL pixel space (loc tokens are resolution-independent).
        model_img = pil
        if self.max_image_edge and max(pil.size) > self.max_image_edge:
            model_img = pil.copy()
            model_img.thumbnail((self.max_image_edge, self.max_image_edge))
            logger.info("  downscaled %dx%d -> %dx%d for speed",
                        orig_w, orig_h, model_img.size[0], model_img.size[1])
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": self.prompt}],
        }]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[model_img], return_tensors="pt").to(self.device)

        prompt_len = inputs["input_ids"].shape[1]
        logger.info("  generating (image %dx%d, %d input tokens, up to %d new)...",
                    model_img.size[0], model_img.size[1], prompt_len, self.max_new_tokens)
        t0 = time.perf_counter()
        with self._torch.no_grad():
            generated = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, use_cache=True)
        n_new = generated.shape[1] - prompt_len
        logger.info("  generated %d tokens in %.1fs (%.1f tok/s)",
                    n_new, time.perf_counter() - t0, n_new / max(time.perf_counter() - t0, 1e-6))
        output = self.processor.batch_decode(
            generated[:, prompt_len:], skip_special_tokens=False)[0].lstrip()

        self.last_raw = output
        if self.raw_dir:
            (self.raw_dir / f"frame_{index:06d}.screentag.txt").write_text(
                output, encoding="utf-8")

        # Boxes in the original resolution, not the downscaled one.
        elements = parse_screentag(output, orig_w, orig_h)
        # ScreenVLM omits typed field values; OCR them back in from the full-res
        # crop so the IDM can see `type` actions.
        if self.ocr is not None:
            before = sum(1 for e in elements if e.editable and e.value)
            ocr_fill_values(elements, pil, self.ocr)
            after = sum(1 for e in elements if e.editable and e.value)
            if after > before:
                logger.info("  OCR filled %d field value(s)", after - before)
        return ScreenState(index=index, ms=ms, elements=elements)
