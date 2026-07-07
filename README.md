# workflow-learning-agents

Workflow Memory: Learning Reusable Procedures from Agent Trajectories

For the big-picture map of how the pieces fit together — the demo→skill pipeline
and the two loops (self-healing repair and the training bridge) — see
`[docs/architecture.md](docs/architecture.md)`.

## Demo2Skill v0

This repo is starting with Module 1 from the Demo2Skill plan: browser-only
manual trace capture with Playwright. The recorder launches a headed browser,
captures human clicks, typing, selected text, URL changes, DOM snapshots,
screenshots, and a Playwright trace, then writes a raw `trace.json`.

Install local dependencies:

```bash
uv sync
uv run playwright install chromium
```

Record the first GitHub issue-creation demo:

```bash
uv run demo2skill-record https://github.com/user/repo \
  --output runs/github_issue_demo
```

In the opened browser, do the demo: open Issues, click New Issue, fill title and
body, and stop before submitting. Return to the terminal and press Enter. The
recorder will save:

- `runs/github_issue_demo/trace.json`
- `runs/github_issue_demo/screens/*.png`
- `runs/github_issue_demo/dom/*.html`
- `runs/github_issue_demo/playwright_trace.zip`

Normalize the raw trace into semantic events:

```bash
uv run demo2skill-normalize runs/github_issue_demo/trace.json
```

This writes `runs/github_issue_demo/semantic_trace.json` with events such as
`navigate`, `click`, `fill_field`, `upload_file`, `select_option`, and
`set_checked`. The normalizer is rule-based for v0: it prefers DOM role,
label/aria-label, placeholder, selected/visible element text, input name/id,
nearby text when available, and selector metadata.

## Induce a workflow skill (Modules 3–5)

Turn a semantic trace into a reusable, parameterized **workflow skill**:

```bash
uv run demo2skill-induce runs/github_issue_demo/semantic_trace.json \
  -o demo2skill/examples/github_issue/induced_workflow.yaml
```

The inducer cleans demonstration noise and generalizes the task:

- **Segmenter** (`induction/segmenter.py`) drops hidden environment writes,
turns login interactions into a `user_logged_in` precondition, collapses
per-keystroke `fill_field` runs to the final committed value, removes focus
clicks, and folds the navigation chain before the form into a single
`navigate` step.
- **Variable abstraction** (`induction/variable_abstraction.py`) lifts each
typed constant into a named input (`"42.50"` → `${amount}`) with an inferred
type (`string` / `number` / `file` / `boolean`).
- **Workflow generator** (`induction/workflow_generator.py`) emits semantic
steps, appends `verify` checks over the filled fields, and adds a
`request_user_confirmation` gate before any irreversible submit.

The pipeline is **pluggable**. The default path is fully deterministic and
needs no API key. To use the LLM path (Anthropic, with prompt caching), install
the extra and pass `--llm`:

```bash
uv sync --extra llm
ANTHROPIC_API_KEY=... uv run demo2skill-induce ... --llm
```

If the model returns invalid YAML, induction falls back to the deterministic
baseline.

### Workflow schema, validation, and storage (Module 5)

Generated YAML is validated by a Pydantic schema (`workflow/schema.py`) that
rejects unknown actions, missing/duplicate step IDs, and malformed targets.
Semantic checks (`workflow/validator.py`) flag **unbound variables**, **unused
inputs**, and **ungated irreversible submits**. `WorkflowStore`
(`workflow/store.py`) persists validated skills as `{workflow_id}.yaml`. A
worked example lives in `demo2skill/examples/github_issue/`.

## Learn from a raw tutorial video (VIDEO2ACTION)

The recorder needs an instrumented browser. To learn from an ordinary
screen-recorded tutorial instead, `demo2skill/video/` provides **two
interchangeable engines** in their own subpackages, both emitting the shared
`Trajectory` (`video/schema.py`) that serializes to a normalize-ready
`trace.json`:

- `video/video2action/` — Engine 1, the VIDEO2ACTION frame engine (below).
- `video/statediff/` — Engine 2, state-diff inverse dynamics (further below),
recommended for GUI demonstrations.



### Engine 1 — VIDEO2ACTION (frame-based)

`video/video2action/` implements a VIDEO2ACTION inverse-dynamics module after
VideoAgentTrek
([arXiv:2510.19488](https://arxiv.org/abs/2510.19488)): a two-stage pipeline that
detects GUI actions with temporal boundaries, then recognizes their structured
content (coordinates, typed text) plus the semantic target (button caption /
field label). Its output is a `Trajectory` that serializes into a normalize-ready
`trace.json`, so a video flows through the *existing* induction pipeline and out
comes a parameterized workflow skill.

```bash
# deterministic path (no model): driven by click/keystroke overlays or a sidecar
uv run demo2skill-video2action \
  --events demo2skill/examples/github_issue/video_events.json \
  -o runs/issue_video --induce
```

This writes `trajectory.json`, `trace.json`, `semantic_trace.json`, and an
induced `workflow.yaml` with `title`/`body` lifted into `${inputs}` — the same
skill the instrumented demo produces, learned from video events alone.

The two IDM stages are backend-pluggable (mirroring `induction/llm.py`):

- `ScriptedBackend` (`video/video2action/backends/scripted.py`) is the
no-model path. It reads explicit action records — which tutorials often already
carry as on-screen keystroke/click overlays, chapter markers, or a
hand-authored sidecar — and drives the full chain deterministically (and the
tests).
- `VLMClient` (`video/video2action/backends/vlm.py`) is the seam for a real
grounding / recognition VLM (Qwen-VL, Claude, …). The two prompts live in
`video/video2action/prompts.py`; frame extraction (ffmpeg / a frames
directory) is in `video/video2action/frames.py`. No model is bundled, so the
package imports without one.



### Engine 2 — State-diff inverse dynamics (recommended for GUI)

GUI actions are sparse, localized, and stateful — a checkbox flips in a 20×20
region, one character appears in a field — so whole-frame visual discontinuity is
the wrong boundary signal. The principle here is: **visual change only proposes
candidate moments; element-level before/after state plus cursor evidence
determine the action.** We recover the action that *connects* two parsed screen
states:

```
(S_t, S_{t+1}) --IDM--> action a_t --> affected element e_t --> effect Δs_t
```

The implemented components (`demo2skill/video/statediff/`):

- `state.py` — `ScreenState` / `UIElement`: a frame parsed into elements
(role, bbox, text, value, focused/checked/selected), with `same_state` dedup
that ignores transient noise (blinking cursor, clock, pointer).
- `matching.py` — element tracking across states via weighted IoU + text +
type + label, yielding matched / appeared / disappeared sets.
- `cursor.py` — cursor-centric evidence (dwell, click signature,
element-under-pointer) that resolves ambiguity (click vs keyboard shortcut) and
supplies coordinates.
- `inverse_dynamics.py` — `TransitionProposer` (stage 1: state-delta proposes
moments) and `StateDiffIDM` (stage 3: classify text-entry / toggle / select /
menu / scroll / navigate / focus-move into a structured action with target,
args, precondition, effect, confidence). `StateTrajectoryBuilder` composes a
`Trajectory`.
- `graph.py` — `UIStateGraph`: deduped state nodes with action-labeled edges,
turning one linear demo into a reusable app map.

Given parsed screen states, the IDM recovers
`click Issues → click New issue → type Title → type Body`, attributes the page
transitions to the control under the cursor, and induces the same parameterized
skill. See `tests/test_state_idm.py` and `examples/github_issue/screen_states.json`.

A video-induced skill is a normal `WorkflowSkill`, so it runs under the executor

- repair loop below just like a recorded one.



#### The pixels→state front (`statediff/parser/`)

The step that turns frames into `ScreenState`s is deliberately isolated so the
IDM stays model-free and testable. That slot is now filled and still pluggable:

- `parser/base.py` — the `ScreenParser` protocol, the `parse_frames` driver,
`build_state` / `load_states` / `state_to_dict`, and `ScriptedScreenParser`
(model-free replay of pre-parsed states).
- `parser/screenvlm.py` — `ScreenVLMParser`, the **real**
`[docling-project/ScreenVLM](https://huggingface.co/docling-project/ScreenVLM)`
checkpoint (Idefics3, 316M; [arXiv:2602.14276](https://arxiv.org/abs/2602.14276)).
It emits **ScreenTag** markup (not JSON); `parse_screentag` converts it to
`UIElement`s, rescaling the `[0,500]` location tokens to pixels.
- `parser/vlm.py` — `VLMScreenParser`, the *generic* path: a general
instruct-VLM prompted to emit dense-parse **JSON**, via a `ScreenParserClient`.
- `parser/clients.py` — JSON-path backends behind lazy imports:
`TransformersScreenVLMClient` (a general HF VLM, `screenvlm` extra),
`AnthropicVisionClient` (Claude vision, reuses the `llm` extra), and
`default_screen_parser_client()` (env-driven).
- `parser/prompts.py` — the dense-parse JSON prompt (every visible element,
not just the task-relevant one).
- `parser/ocr.py` — ScreenVLM detects input boxes but, trained on freshly
rendered pages, does **not** transcribe *typed* field values. This OCRs each
editable element's full-res crop to fill `value` — which is what lets the IDM
see `type` actions. Backends: `EasyOCRBackend` (pip-only, `ocr-easy` extra),
`PaddleOCRBackend` (pip-only, `ocr-paddle` extra), and `TesseractOCR` (`ocr`
extra + the system `tesseract` binary). Precision guards: tight inset + upscaled
crops, **label/placeholder echo rejection** (a read that matches a nearby label
or placeholder is discarded), and **one-owner dedup** (overlapping field boxes
can't both claim the same text).
- `statediff/field_text.py` — cleans field values *across frames*, applying the
state-diff idea to text. **Placeholder vs typed value** is separated by
appearance time (a placeholder sits at rest and vanishes on focus; typed text
grows) — the pre-interaction text is tagged `placeholder_text` and cleared from
`value`. **Temporal voting** then fuses the many noisy OCR reads of a typing run
(per-character majority over the near-complete frames) into one clean value,
beating any single-frame read. Runs automatically after parsing; abstains when
the run's start isn't captured.
- `statediff/cursor_detect.py` — the pixels→cursor front: template-matches the
pointer in each frame (`TemplateCursorDetector`) to produce the `CursorTrack`
the IDM uses for clicks and coordinates. `cursor` extra (numpy + OpenCV).

Run the parser front over a recording with `demo2skill-parse-video`:

```bash
# 1. free dry run — no model, no ffmpeg: replay parsed states through the pipeline
uv run demo2skill-parse-video \
  --replay demo2skill/examples/github_issue/screen_states.json \
  -o runs/parse/states.json --trace-out runs/parse/trace.json --graph

# 2. a real recording: ScreenVLM (structure) + OCR (typed values) + cursor
#    detection (clicks). Use MPS on Apple Silicon; float32 avoids fp16
#    instability; downscaling + a token cap keep it fast.
uv sync --extra screenvlm --extra ocr-easy --extra cursor
uv run demo2skill-parse-video demo2skill/examples/github_issue/demo.mov \
  --client screenvlm --ocr easyocr \
  --detect-cursor template --cursor-template demo2skill/templates/cursor/cursor_tight.png \
  --device mps --dtype float32 --image-max-edge 1280 --max-new-tokens 1024 \
  --sample fps --fps 1 --max-frames 40 \
  -o runs/parse/states.json --trace-out runs/parse/trace.json --graph
# pin the training-data version with --revision v1 or --revision v2

# 3. or a general instruct-VLM via Claude vision (one vision call per frame)
uv sync --extra llm
ANTHROPIC_API_KEY=... uv run demo2skill-parse-video demo.mp4 \
  --client anthropic --sample fps --fps 1 --max-frames 40 \
  -o runs/parse/states.json --trace-out runs/parse/trace.json
```

Frame extraction (`video2action/frames.py`, via ffmpeg — any container: mp4,
mov, mkv, webm, …) offers three sampling strategies, chosen with `--sample`:
`fps` (uniform at `--fps` N/s), `keyframes` (encoded I-frames only — cheap, good
for slide-like tutorials), and `scene` (scene-change boundaries above
`--scene-threshold` — adaptive, good for busy UIs). Per-frame timestamps and
resolution are read from ffmpeg/ffprobe, not assumed. Pre-extracted frames can be
passed with `--frames-dir` to skip ffmpeg entirely.

This writes the parsed `states.json` (inspect it first — parser quality gates
everything downstream) and, with `--trace-out`, a normalize-ready `trace.json`
that flows straight into induction. Three complementary fronts turn raw pixels
into the full observation the IDM needs: **ScreenVLM** for structure, **OCR**
(`--ocr`) for the typed field values ScreenVLM omits, and **cursor detection**
(`--detect-cursor`) for click positions. A **temporal field-text pass**
(`statediff/field_text.py`) then runs automatically, separating placeholders from
typed values and majority-voting the OCR reads across each typing run — so bump
`--fps` a little (e.g. `2`) to give it more frames to average over and to catch
the rest→focus→typed transition. Give `--cursor-template` a real crop of your OS
pointer for reliable matching, or pass recorded pointer data with
`--cursor cursor.json` to skip detection entirely.

Performance notes (Apple Silicon): `--device mps` runs on the GPU; `--dtype float32` avoids occasional fp16-on-MPS hallucination; `--image-max-edge` downsizes
Retina screenshots so the model tiles fewer patches; `--max-new-tokens` caps the
decode. The checkpoint ships with `use_cache=False`, which the parser overrides —
without it generation is ~40× slower.

## Execute a skill and self-heal (Modules 6–9)

The reliability core: run an induced skill against a page, verify each step, and
repair the skill when the UI has shifted underneath it.

```bash
uv run demo2skill-run demo2skill/examples/github_issue/induced_workflow.yaml \
  --page demo2skill/examples/github_issue/page_shifted.html \
  --inputs demo2skill/examples/github_issue/test_inputs.json
```

The executor (`executor/executor.py`) binds inputs, grounds each target against
the page, acts, and verifies:

- **Strict grounding** (`executor/grounding.py`) acts only on an exact
identifier match (selector/label/aria/placeholder/text). Anything it can't pin
down is routed to repair rather than clicked blindly.
- **Repair loop** (`executor/repair.py`) re-grounds the stale target
*semantically* (token/fuzzy over role-compatible elements), proposes a minimal
`replace` patch, and **re-validates it against the Pydantic schema + safety
validator** before retrying — so a repair can never remove a confirmation gate
or introduce an unknown action. Bounded retries with an oscillation guard.
- **Verifier** (`executor/verify.py`) reads page state to confirm
`field_equals` / `field_filled` / `page_contains` and step postconditions.

On the matched page the skill grounds by selector with **zero repairs**
(`converged`). On `page_shifted.html` — volatile ids changed, labels reworded —
strict grounding fails, the loop re-grounds and patches `fill_title`/`fill_body`
to the page's current labels, and the run still reaches the confirmation gate.
The repaired (more robust) skill is what a `WorkflowStore` would persist for next
time. Irreversible submits stay gated: the run halts at
`request_user_confirmation` unless `--yes` is passed.

## Export verified trajectories for training (the bridge)

The same run that drives repair is also a source of *verified* supervision. The
executor + verifier confirm which action actually worked on which element —
something raw video-mining pipelines lack — so `demo2skill/export/` turns a run
into a training-ready `trajectory.jsonl` in the `(observation, instruction, action)` shape a policy model consumes, keeping **only** the steps the run
confirmed.

```bash
uv run demo2skill-export demo2skill/examples/github_issue/induced_workflow.yaml \
  --page demo2skill/examples/github_issue/page_match.html \
  --inputs demo2skill/examples/github_issue/test_inputs.json \
  -o runs/github_issue_demo/trajectory.jsonl
```

Each JSONL line is one confirmed step: `${title}` is bound to the concrete typed
string, the exported target is the *repaired* (not the brittle demo) locator, and
`verified` / provenance record how it was grounded. Halting at the confirmation
gate counts as a good episode, so no irreversible action is taken to produce
data. This is the seam that lets an editable, verified skill double as filtered
supervision — see `[docs/architecture.md](docs/architecture.md)`.

### Run the tests

```bash
uv run python -m unittest discover -s tests
```

