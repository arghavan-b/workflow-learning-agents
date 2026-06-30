# workflow-learning-agents

Workflow Memory: Learning Reusable Procedures from Agent Trajectories

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

- **`video/video2action/`** — Engine 1, the VIDEO2ACTION frame engine (below).
- **`video/statediff/`** — Engine 2, state-diff inverse dynamics (further below),
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

- **`ScriptedBackend`** (`video/video2action/backends/scripted.py`) is the
  no-model path. It reads explicit action records — which tutorials often already
  carry as on-screen keystroke/click overlays, chapter markers, or a
  hand-authored sidecar — and drives the full chain deterministically (and the
  tests).
- **`VLMClient`** (`video/video2action/backends/vlm.py`) is the seam for a real
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

    (S_t, S_{t+1}) --IDM--> action a_t --> affected element e_t --> effect Δs_t

The implemented components (`demo2skill/video/statediff/`):

- **`state.py`** — `ScreenState` / `UIElement`: a frame parsed into elements
  (role, bbox, text, value, focused/checked/selected), with `same_state` dedup
  that ignores transient noise (blinking cursor, clock, pointer).
- **`matching.py`** — element tracking across states via weighted IoU + text +
  type + label, yielding matched / appeared / disappeared sets.
- **`cursor.py`** — cursor-centric evidence (dwell, click signature,
  element-under-pointer) that resolves ambiguity (click vs keyboard shortcut) and
  supplies coordinates.
- **`inverse_dynamics.py`** — `TransitionProposer` (stage 1: state-delta proposes
  moments) and `StateDiffIDM` (stage 3: classify text-entry / toggle / select /
  menu / scroll / navigate / focus-move into a structured action with target,
  args, precondition, effect, confidence). `StateTrajectoryBuilder` composes a
  `Trajectory`.
- **`graph.py`** — `UIStateGraph`: deduped state nodes with action-labeled edges,
  turning one linear demo into a reusable app map.

Given parsed screen states (what a screen parser such as OmniParser/ScreenParse
emits — the pixels→state step is the pluggable front), the IDM recovers
`click Issues → click New issue → type Title → type Body`, attributes the page
transitions to the control under the cursor, and induces the same parameterized
skill. See `tests/test_state_idm.py` and `examples/github_issue/screen_states.json`.

A video-induced skill is a normal `WorkflowSkill`, so it runs under the executor
+ repair loop below just like a recorded one.

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

### Run the tests

```bash
uv run python -m unittest discover -s tests
```