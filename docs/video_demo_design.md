# Design: Video Demonstrations + Repair Loop for Demo2Skill

Status: proposal · Scope: Modules 1–9 extension · Author: design draft

## 1. What changes and why

Today the demonstration source is an **instrumented Playwright recorder**. It
hands the pipeline a rich raw trace: CSS selectors, DOM snapshots, the
accessibility tree, element `name`/`id`/`role`, and the exact typed text. Every
stage downstream — `trace/normalize.py`, `induction/*`, `workflow/schema.py` —
was built assuming that metadata is present.

This design replaces the recorder with a **screen-recording video**. The demo is
just pixels (optionally plus a cursor, keystroke overlay, and narration audio).
There is no DOM, no selector, no accessibility tree, and no ground-truth typed
string. Everything the old recorder read directly from the page must now be
**reconstructed from frames**.

The design goal is to change *only the front of the pipeline*. Video ingestion
produces a raw `trace.json` that is schema-compatible with what the Playwright
recorder emitted (minus the fields video can't provide), so `normalize.py` and
the entire induction stack keep working unchanged. Then it adds the missing
back half of the system — executor, verifier, and the **repair/refinement
loop** — which the repo does not yet have (`demo2skill/executor/` and
`demo2skill/eval/` do not exist yet).

```
Video (mp4 + optional audio)
  ↓  [NEW] Video Ingestion
raw trace.json   ← same shape as recorder output, selector/dom fields absent
  ↓  trace/normalize.py            (unchanged)
semantic_trace.json
  ↓  induction/ (segmenter → variable_abstraction → workflow_generator)  (unchanged)
workflow.yaml  → workflow/schema.py + validator.py + store.py            (unchanged)
  ↓  [NEW] Executor + Target Grounder + Verifier
  ↺  [NEW] Repair Loop  → store v2, vN
```

## 2. Why video is hard (and what survives)

| Field the recorder gave us | Available from video? | Replacement |
| --- | --- | --- |
| `selector` (`input#_r_22_`) | No | omit — downstream already tolerates missing selectors |
| `dom_snapshot` | No | per-frame screenshot + OCR text layer |
| `element.role` | Inferred | VLM / UI-element detector |
| `element.label`, `aria_label`, `placeholder` | Inferred | OCR of nearby label text |
| `target_text` (button caption) | Yes (OCR) | OCR under cursor |
| `typed_text` (exact value) | Approx (OCR) | OCR the field's final value |
| `url`, `page_title` | Approx (OCR) | OCR the address bar + window title |
| `mouse {x,y}` | Yes if cursor visible | cursor tracking |

The important compatibility fact: `workflow/schema.py`'s `Target` only requires
**one** identifier out of `text, label, role, selector, aria_label,
placeholder, nearby_text, semantic` (or a `url`). A video-derived target that
carries `{text, label, role}` and no selector already validates. Likewise
`normalize.py` uses `drop_empty(...)`, so absent `selector`/DOM fields simply
vanish rather than break. **No schema change is required to accept
video-derived traces** — only additive fields (Section 7).

## 3. Video ingestion (the new front module)

Reading a video splits into **two layers** that should not be conflated:

- **Coarse / temporal layer** — *where* in time things happen and *what kind* of
  thing each segment is. This is exactly what video semantic-embedding models
  (e.g. Twelve Labs **Marengo** for embeddings/moment-localization, **Pegasus**
  for clip Q&A) are good at. Embeddings localize the moment; they do not read
  the frame.
- **Fine / grounding layer** — *which* element, the verbatim label, the exact
  typed value, the URL. An embedding vector compresses a clip into gist and
  discards this pixel-level text by design, so this layer stays OCR + UI-element
  detection + single-frame VLM. You cannot recover "the textbox now contains
  `Bug in login flow`" from a clip embedding.

The two layers are complementary: embeddings tell `action_detector` *when* to
look, frame reading tells `ui_reader` *what's there*.

New package `demo2skill/video/`:

```
video/
  ingest.py          # CLI: video → raw trace.json
  frames.py          # decode, sample, scene-cut detection
  embed.py           # coarse layer: Marengo embeddings, moment localization
  action_detector.py # embedding moments + frame deltas + cursor → action events
  ui_reader.py       # fine layer: OCR + element detection + VLM (or Pegasus)
  cursor.py          # cursor tracking (optional keystroke-overlay parsing)
  narration.py       # optional audio transcript → intent hints
```

### 3.1 Frame sampling and action segmentation

A video is a continuous stream; the pipeline needs **discrete action events**.
The detector fuses two signals: **embedding-based moment localization** (the
coarse layer — "a form-fill happens around 0:12–0:18", "a submit/navigation
moment here") to propose action boundaries, refined by **cursor dynamics and
frame deltas** for precise frame selection. Embeddings are a stronger, less
heuristic boundary signal than frame-delta thresholds alone, especially on busy
UIs with animation and video content. Action types:

- **Click** — cursor decelerates and settles, a brief press animation or focus
  ring appears, and a localized UI change follows. Emit one `click` at the
  settle frame; the target region is the area under the cursor.
- **Type** — text grows character-by-character inside a focused field across
  consecutive frames. Collapse the whole run into one `type` event; the value is
  the field contents in the **last** frame of the run (mirrors how
  `normalize.py` already collapses per-keystroke `fill_field` runs).
- **Navigate** — the address bar text changes or the whole viewport replaces
  (large global frame delta). Emit `navigation` with the OCR'd URL.
- **Scroll** — vertical content translation with a static cursor; usually
  non-essential, recorded but later dropped by the segmenter.
- **Select option / toggle** — dropdown opens then a row is chosen; checkbox or
  radio glyph flips state.

Each detected action becomes a raw event positioned by `frame_index` /
`frame_time_ms` instead of the recorder's wall-clock `timestamp`.

### 3.2 Reading the UI for each action

For every action event, `ui_reader.py` extracts the semantic target from the
frame using a two-tier strategy (cheap first, VLM as fallback) to control cost:

1. **OCR layer** over the action's local region: the control's visible caption,
   the nearest label text to its left/top, placeholder text, and the address
   bar / window title.
2. **UI-element detection** (a layout/element model, or a VLM prompt) to assign
   a `role` (button, textbox, combobox, checkbox, link) and a bounding box.
3. **VLM fallback** only when 1–2 are low-confidence: "What control is the
   cursor interacting with, and what is its label?" returns `{role, label,
   text, bbox, confidence}`.

For `type` events the field's **final OCR value** becomes `typed_text`. This is
the single biggest accuracy risk (Section 8) and the main thing the repair loop
later corrects.

### 3.3 Narration → intent (optional)

If the demo has a voice-over ("now I open a new issue and paste the title"), the
transcript is aligned to actions by timestamp and attached as `intent_hint` on
events / segments. The existing LLM segmenter and `induction/prompts.py` can use
these hints; the deterministic baseline ignores them.

### 3.4 Coarse layer: segment intent and retrieval

Beyond boundary detection, the embedding layer feeds two downstream uses:

- **Segment intent labeling** — embed each detected segment and classify it
  (`login` / `open-form` / `fill-form` / `review` / `submit`). This is a cheap,
  non-LLM signal the segmenter can use alongside its current URL/title/form
  heuristics, and it populates `Segment.intent` without a generation call.
- **Demo retrieval against memory** — embed the whole demo and query the
  workflow store for prior skills covering the same task ("have we learned this
  before?"), so a new video can refine an existing skill instead of creating a
  near-duplicate.

### 3.5 Output

`ingest.py` writes a raw `trace.json` with `metadata.source_modality:
"video"`. From here the existing commands run unchanged:

```bash
demo2skill-video-ingest demo.mp4 -o runs/issue_video
demo2skill-normalize     runs/issue_video/trace.json
demo2skill-induce        runs/issue_video/semantic_trace.json -o workflow.yaml
```

## 4. Multi-demo alignment

A single demo conflates the **causal** steps of a task with **incidental** ones —
the particular value typed, a stray exploratory click, one specific path through
the UI. You cannot tell which is which from one trace. Several demos of the
*same task with different inputs/paths* make the invariant structure observable,
and the coarse embedding layer makes aligning them cheap. This is the strongest
lever in the whole design for robustness, and it reuses the embedding
infrastructure already introduced for segmentation.

New module `demo2skill/induction/aligner.py`, sitting **between per-demo
ingestion and induction**:

```bash
demo2skill-align runs/issue_v1/semantic_trace.json \
                 runs/issue_v2/semantic_trace.json \
                 runs/issue_v3/semantic_trace.json \
  -o runs/issue_consolidated/semantic_trace.json
```

It emits a single consolidated `semantic_trace.json` annotated with per-step
support, so the existing `induce` pipeline consumes it unchanged.

### 4.1 Segment correspondence

Embed each demo's segments (Marengo) and align the segment sequences across
demos by embedding similarity — a sequence alignment (DTW-style) over segment
vectors, not a positional zip, so it tolerates extra/missing/reordered steps.
The output is a correspondence: "fill-title in demo A ≈ fill-title in demo B ≈
fill-title in demo C." The fine layer (role + label) confirms identity where
embeddings are ambiguous.

### 4.2 Causal intersection

With segments aligned, support counting replaces the segmenter's current
heuristic `essential` flag with **evidence**:

- A step present in **all / most** demos is causal → keep, mark `essential`.
- A step present in **one** demo is incidental → drop (the exploratory click,
  the accidental scroll).
- Ordering that varies across demos marks a step as **order-independent**;
  steps that are sometimes absent become **optional** / branch variants. v0
  keeps the majority path and records the others for the recovery layer.

### 4.3 Variability → variables (free, stronger abstraction)

This is where alignment pays off twice. For a corresponded step, look at the
**value across demos**:

- Value **differs** every demo (`"Bug in login"`, `"Typo in header"`, …) →
  it's a parameter. Lift to `${input}` with high confidence — far stronger than
  the single-demo heuristic in `variable_abstraction.py`, which has to guess
  from one example.
- Value **constant** across demos → it's a genuine literal; keep it.

So alignment directly improves variable abstraction instead of competing with
it.

### 4.4 Uncertainty → active learning

When alignment is ambiguous (a step seen only once, low correspondence
confidence, or a value that's constant across too few demos to trust), the
system **requests another demo** rather than guessing — bounded by a max-demos
budget. This is the multi-demo analogue of the repair loop: each new demo is a
cheap signal that sharpens the induced skill before any execution happens.

The consolidated trace carries, per step, `support` (how many demos),
`order_stable`, and `value_varies` flags. Induction reads these to set
`essential`, choose variables, and emit branch/recovery hints.

## 5. Executor, grounding, verifier (Modules 6–8)

These don't exist yet and are required before a repair loop has anything to
repair. The demonstration modality (video) is **decoupled** from the execution
substrate — a skill learned from a video can run two ways:

- **DOM-grounded execution** (Playwright): the semantic target
  (`label`/`role`/`text`) is resolved against the live DOM. Preferred when the
  target app is a web page, because it is robust to layout shift.
- **Pixel-grounded execution** (VLM → click coordinates): for apps that can't be
  instrumented (native/desktop, canvas UIs). The grounder takes the current
  screenshot + the step's semantic target and returns a click point.

### 5.1 Target grounder

Because video traces carry **no selector**, the grounder cannot use the
recorder's "exact selector first" shortcut. Ordered strategy:

1. text / label exact match (DOM or OCR)
2. aria-label / placeholder match
3. role + nearby-text match
4. `grounding_hint` region from the demo (normalized bbox) as a spatial prior
5. VLM/visual match as fallback

It returns `{locator, confidence, grounding_method}`. Below a confidence
threshold the step does **not** execute blindly — it raises
`low_grounding_confidence`, which the repair loop handles.

### 5.2 Verifier

Rule-based first, VLM fallback — never everything-LLM (it's flaky). The
`workflow/schema.py` check vocabulary already covers the v0 needs:
`field_equals`, `field_filled`, `page_contains`, plus step `postcondition`
(`page_contains` / `url_contains`). On a video substrate "field value" is read
by OCR rather than DOM `.value`, so `field_equals` tolerates fuzzy/normalized
string comparison.

## 6. The repair / refinement loop (the "loops" ask)

This is the heart of the request. A learned workflow is a hypothesis; the loop
turns failed executions into a corrected, versioned skill. It operates at two
time scales.

### 6.1 Inner loop — within a single run

```
for step in workflow.steps:
    bind_variables(step, inputs)
    locator = ground_target(step.target, page)      # may fail / be low-confidence
    result  = execute_action(step, locator, value)
    ok      = verify(step, page)
    if not ok:
        record  = make_failure_record(step, page)
        patch   = propose_repair(record, workflow)   # rule first, LLM fallback
        if patch and within_budget(step):
            workflow = apply_and_revalidate(workflow, patch)  # Pydantic re-check
            retry step                                # bounded: e.g. 3 attempts
        else:
            run_recovery(step)                        # ask_user / stop_and_report
```

**Failure record** (extends the doc's Module 9 object with video context):

```json
{
  "workflow_id": "create_github_issue_v1",
  "step_id": "fill_title",
  "attempt": 1,
  "failure_type": "target_not_found",
  "previous_target": { "label": "Add a title", "role": "textbox" },
  "grounding_confidence": 0.34,
  "page_snapshot": "screens/fail_fill_title_01.png",
  "page_text": "<OCR/DOM text dump>",
  "error": "No matching element found"
}
```

**Repair is keyed by `failure_type`, and the keys are exactly the `recovery`
rules already emitted into every workflow** (`target_not_found`,
`missing_input`, `verification_failed`):

| failure_type | strategy | patch outcome |
| --- | --- | --- |
| `target_not_found` / `low_grounding_confidence` | broaden semantic search, then VLM re-ground | `replace` step `target` with a better-matching locator |
| `postcondition_failed` | a dialog/extra screen appeared | `insert` a step (dismiss/wait) before continuing |
| `verification_failed` | data is wrong, not the path | `stop_and_report` — never silently "fix" user data |
| `missing_input` | required input unbound | `ask_user` |
| `action_error` / `timeout` | transient / not-yet-ready | wait + retry, no patch |

**Patch shape** (validated by `WorkflowSkill` before it is applied, so a bad
repair can never reach the runtime loop):

```yaml
op: replace            # replace | insert | delete
step_id: fill_title
target:
  label: "Title"       # was "Add a title"
  role: textbox
```

### 6.2 Loop safety

A repair loop that can edit its own program needs guard rails:

- **Bounded attempts** per step (default 3) and per run (total repair budget).
- **Oscillation guard** — never re-apply a patch whose signature already failed
  this run; keep a tried-patch set.
- **Re-validation gate** — every patched workflow must pass `workflow/schema.py`
  + `workflow/validator.py` (unbound vars, ungated submits) before retry.
- **No-fabrication rule** — `verification_failed` and irreversible actions are
  never auto-repaired; they escalate to the user. The
  `request_user_confirmation` gate before submit is preserved across repairs.

### 6.3 Outer loop — refinement across runs

Single-run repairs are local. The refinement loop accumulates them to make the
**stored skill** more robust over time (this is the paper's "procedural memory
improves robustness" claim):

1. Run the induced workflow against a set of perturbed variants (eval harness).
2. Collect failure → repair pairs across runs.
3. **Promote** repairs that recur (e.g. the demo's label `"Add a title"` keeps
   losing to `"Title"`) into the canonical target; **demote** brittle locators
   (drop the volatile `selector#_r_22_`, keep `role`+`label`).
4. Persist as a new version via `WorkflowStore` (`workflow_id` `…_v2`), with
   `derived_from` provenance and a repair log, never overwriting v1.
5. A run that completes with **zero** repairs marks the version *converged*.

This closes the learning loop: video demo → induced v1 → executed → repaired →
v2, with each version measurably less brittle than the last.

## 7. Data-object additions

All additive — nothing in the current schemas is removed.

**Raw event (video):** add `source_modality`, `frame_index`, `frame_time_ms`,
`cursor {x,y}`, `ocr_text`, `detected_elements[]`, `grounding_hint {bbox,
confidence}`. Drop (leave absent) `selector`, `dom_snapshot`,
`accessibility_tree`.

**Video demo metadata:** `fps`, `resolution`, `duration_ms`, `has_cursor`,
`narration_transcript`.

**Workflow step (optional):** `grounding_hint` (normalized demo bbox) as a
spatial prior for the grounder; `version` / `derived_from` on `WorkflowSkill`
for provenance. These are optional and default-absent, so existing v0 YAML
stays valid.

**New runtime objects:** `FailureRecord`, `RepairPatch`, `RepairLog`,
`RunResult` (in a new `demo2skill/executor/`).

## 8. Risks and mitigations

- **Wrong typed value (OCR).** The biggest video-specific failure. Mitigation:
  variable abstraction lifts typed constants into `${inputs}` anyway, so the
  exact demo string matters less; the verifier's `field_equals` catches
  mismatches and the loop reports rather than guesses.
- **Action-boundary ambiguity** (was that two clicks or one?). Mitigation:
  cursor-settle + UI-change confirmation, and the segmenter already drops
  duplicate/exploratory actions.
- **Invisible cursor / keystroke overlay absent.** Fall back to frame-delta
  localization; recommend (don't require) cursor-highlight when recording.
- **VLM cost/latency.** Two-tier reading (OCR/detector first, VLM only on
  low-confidence) bounds calls; grounding hints cache spatial priors.
- **Privacy.** Screen recordings can capture secrets; ingestion should redact
  password-type fields (no value captured) and keep frames local.
- **Hosted embedding dependency / reproducibility.** The coarse layer (Marengo)
  is a hosted, versioned API — Marengo 2.7 was sunset on 2026-03-30, after which
  previously indexed embeddings could no longer be retrieved. For a benchmark
  artifact this is a reproducibility hazard: pin the model version, **cache
  embeddings to disk**, and keep an open-model fallback for the segmentation
  signal so the pipeline degrades gracefully rather than breaking.
- **Too few demos for alignment.** Causal intersection needs ≥2–3 demos to
  separate signal from incident; with one demo the system falls back to
  single-demo heuristics and flags low confidence (Section 4.4).

## 9. Build order

1. `video/ingest.py`: video → frames → action events → raw `trace.json`
   (heuristic detector, OCR only). Prove `normalize` + `induce` run end-to-end on
   it, reusing the GitHub-issue task.
2. Add `video/embed.py` (coarse layer): embedding moment-localization for action
   boundaries + segment-intent labels. Add VLM/element reading for robust
   `role`/`label` and address-bar URL.
3. `induction/aligner.py`: multi-demo alignment → consolidated trace with
   `support` / `value_varies` flags feeding `essential` + variable abstraction.
4. `executor/` + `target_grounder` (DOM mode first) + rule-based `verifier`.
5. Inner repair loop with bounded retries and patch re-validation.
6. `eval/` harness: raw replay vs induced vs induced+verifier vs
   induced+repair, on perturbed variants.
7. Outer refinement loop: promote/demote, versioned store, convergence metric.

First end-to-end milestone is unchanged from the original plan — **create a
GitHub issue, stop before submit** — but now driven from a screen recording
instead of the instrumented recorder.

## 10. Where procedural memory lives

This whole design is an instance of *Selective Procedural Imitation*. Procedural
memory is not a single module — it is the distinction that organizes the
pipeline, so it is worth stating where it lives.

**Episodic vs. procedural.** The raw / semantic trace is *episodic* memory: one
specific time a human did the task, full of incidental detail. Replaying it is
the brittle script the framework rejects. The induced **workflow skill YAML is
the procedural memory** — the durable, parameterized, semantically-targeted "how
to do X," stripped of the episode's accidents. `WorkflowStore` is the procedural
memory store. The pipeline halves are therefore memory operations:
ingest → align → induce is **acquisition + consolidation**; executor → repair →
refinement is **retrieval + use + updating** — acquired from watching,
consolidated by repetition, refined by practice, executed without re-deriving.

**The framework's claims, mapped to mechanisms:**

| Claim | Mechanism in this design |
| --- | --- |
| infers latent **subgoals** | segmenter (named segments + `intent`); each subgoal's success encoded as `postcondition` / `verify` check |
| identifies **causally relevant** actions (the *Selective* part) | multi-demo causal intersection (§4.2); single-demo, the segmenter drops login/exploration/duplicates |
| induces a **reusable** procedural memory | variable abstraction + workflow generator → parameterized YAML; stored + versioned |
| robust to **layout shift** | semantic targets (label/role/text, not selectors) + grounder + repair |
| robust to **goal parameter changes** | `${inputs}` from variable abstraction |
| robust to **interface perturbations** | verifier + repair loop + outer refinement (promote durable locators, demote brittle ones) |

**To fully earn "memory" (not just "a parameterized script")** the store needs
two things this design only seeds: **retrieval / indexing** (the embedding-based
demo retrieval in §3.4 is the start — find the right skill for a new situation)
and **hierarchical composition** (latent subgoals becoming reusable *sub-skills*
that larger skills call). Today `WorkflowStore` is mostly persistence; these two
extensions are what turn a library of skills into procedural memory.
