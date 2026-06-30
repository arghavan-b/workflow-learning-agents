"""Manual browser demonstration recorder built on Playwright.

This is Module 1 from the Demo2Skill plan: capture a human browser task and
save a raw `trace.json` with screenshots, DOM snapshots, and Playwright tracing.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from demo2skill.recorder.dom_snapshotter import DomSnapshotter
from demo2skill.recorder.event_logger import EventLogger, compact_element_info
from demo2skill.recorder.screenshot_store import ScreenshotStore


RECORDER_SCRIPT = r"""
(() => {
  if (window.__demo2skillRecorderInstalled) return;
  window.__demo2skillRecorderInstalled = true;
  window.__demo2skillEvents = window.__demo2skillEvents || [];
  window.__demo2skillPendingInputs = window.__demo2skillPendingInputs || new Map();

  const textOf = (node) => (node && (node.innerText || node.textContent || "") || "")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 500);

  const cssEscape = (value) => {
    if (window.CSS && window.CSS.escape) return window.CSS.escape(value);
    return String(value).replace(/["\\]/g, "\\$&");
  };

  const roleFor = (el) => {
    const explicit = el.getAttribute && el.getAttribute("role");
    if (explicit) return explicit;
    const tag = (el.tagName || "").toLowerCase();
    const type = (el.getAttribute && (el.getAttribute("type") || "").toLowerCase()) || "";
    if (tag === "button" || type === "button" || type === "submit") return "button";
    if (tag === "a") return "link";
    if (tag === "textarea") return "textbox";
    if (tag === "select") return "combobox";
    if (tag === "input") {
      if (type === "checkbox") return "checkbox";
      if (type === "radio") return "radio";
      return "textbox";
    }
    return null;
  };

  const selectorFor = (el) => {
    if (!el || !el.tagName) return null;
    const tag = el.tagName.toLowerCase();
    const id = el.getAttribute("id");
    if (id) return `${tag}#${cssEscape(id)}`;
    const name = el.getAttribute("name");
    if (name) return `${tag}[name="${cssEscape(name)}"]`;
    const aria = el.getAttribute("aria-label");
    if (aria) return `${tag}[aria-label="${cssEscape(aria)}"]`;

    const parts = [];
    let cur = el;
    while (cur && cur.nodeType === Node.ELEMENT_NODE && parts.length < 5) {
      const curTag = cur.tagName.toLowerCase();
      const curId = cur.getAttribute("id");
      if (curId) {
        parts.unshift(`${curTag}#${cssEscape(curId)}`);
        break;
      }
      let index = 1;
      let sibling = cur;
      while ((sibling = sibling.previousElementSibling)) {
        if (sibling.tagName === cur.tagName) index += 1;
      }
      parts.unshift(`${curTag}:nth-of-type(${index})`);
      cur = cur.parentElement;
    }
    return parts.join(" > ");
  };

  const labelFor = (el) => {
    if (!el) return null;
    const aria = el.getAttribute("aria-label");
    if (aria) return aria.trim();
    const labelledBy = el.getAttribute("aria-labelledby");
    if (labelledBy) {
      const label = labelledBy
        .split(/\s+/)
        .map((id) => textOf(document.getElementById(id)))
        .filter(Boolean)
        .join(" ");
      if (label) return label;
    }
    if (el.id) {
      const labelEl = document.querySelector(`label[for="${cssEscape(el.id)}"]`);
      if (labelEl) return textOf(labelEl);
    }
    const wrappingLabel = el.closest && el.closest("label");
    if (wrappingLabel) return textOf(wrappingLabel);
    return el.getAttribute("placeholder") || el.getAttribute("name") || null;
  };

  const elementInfo = (el) => {
    if (!el || el.nodeType !== Node.ELEMENT_NODE) return {};
    const rect = el.getBoundingClientRect();
    const tag = (el.tagName || "").toLowerCase();
    const type = el.getAttribute("type");
    const isPassword = tag === "input" && String(type || "").toLowerCase() === "password";
    return {
      selector: selectorFor(el),
      tag,
      type,
      role: roleFor(el),
      label: labelFor(el),
      text: textOf(el),
      aria_label: el.getAttribute("aria-label"),
      placeholder: el.getAttribute("placeholder"),
      name: el.getAttribute("name"),
      id: el.getAttribute("id"),
      value: isPassword ? null : (("value" in el) ? el.value : null),
      bounding_box: {
        x: Math.round(rect.x),
        y: Math.round(rect.y),
        width: Math.round(rect.width),
        height: Math.round(rect.height)
      }
    };
  };

  const push = (event) => {
    window.__demo2skillEvents.push({
      timestamp: new Date().toISOString(),
      url: location.href,
      page_title: document.title,
      selected_text: String(window.getSelection ? window.getSelection() : "").slice(0, 1000),
      ...event
    });
  };

  const flushInput = (key) => {
    const pending = window.__demo2skillPendingInputs.get(key);
    if (!pending) return;
    clearTimeout(pending.timer);
    window.__demo2skillPendingInputs.delete(key);
    push(pending.event);
  };

  window.__demo2skillFlushPendingInputs = () => {
    Array.from(window.__demo2skillPendingInputs.keys()).forEach(flushInput);
  };

  window.__demo2skillDrainEvents = () => {
    window.__demo2skillFlushPendingInputs();
    const events = window.__demo2skillEvents.slice();
    window.__demo2skillEvents.length = 0;
    return events;
  };

  document.addEventListener("click", (event) => {
    const el = event.target;
    const info = elementInfo(el);
    push({
      action_type: "click",
      selector: info.selector || null,
      target_text: info.text || null,
      target_label: info.label || null,
      element: info,
      mouse: { x: Math.round(event.clientX), y: Math.round(event.clientY) },
      keyboard_text: null
    });
  }, true);

  document.addEventListener("input", (event) => {
    const el = event.target;
    const info = elementInfo(el);
    const key = info.selector || `${info.tag}:${info.name || info.id || info.label || "unknown"}`;
    const existing = window.__demo2skillPendingInputs.get(key);
    if (existing) clearTimeout(existing.timer);
    const record = {
      action_type: "type",
      selector: info.selector || null,
      target_text: info.text || null,
      target_label: info.label || null,
      typed_text: info.value,
      keyboard_text: info.value,
      element: info,
      mouse: null
    };
    const timer = setTimeout(() => flushInput(key), 650);
    window.__demo2skillPendingInputs.set(key, { event: record, timer });
  }, true);

  document.addEventListener("change", (event) => {
    const info = elementInfo(event.target);
    push({
      action_type: "change",
      selector: info.selector || null,
      target_text: info.text || null,
      target_label: info.label || null,
      value: info.value,
      element: info,
      mouse: null,
      keyboard_text: null
    });
  }, true);
})();
"""


def load_playwright() -> Any:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SystemExit(
            "Playwright is required for recording. Install it with:\n"
            "  python3 -m pip install -e .\n"
            "  python3 -m playwright install chromium"
        ) from exc
    return sync_playwright


def page_context(page: Any) -> Dict[str, Optional[str]]:
    try:
        title = page.title()
    except Exception:
        title = None
    return {"url": getattr(page, "url", None), "page_title": title}


def drain_browser_events(page: Any) -> List[Mapping[str, Any]]:
    try:
        events = page.evaluate(
            "() => window.__demo2skillDrainEvents ? window.__demo2skillDrainEvents() : []"
        )
    except Exception:
        return []
    return events or []


def install_recorder_script(page: Any) -> None:
    try:
        page.evaluate(RECORDER_SCRIPT)
    except Exception:
        # The add_init_script path will cover normal navigations; this is only
        # for attaching to the initial page before/after a load.
        pass


def artifact_paths(
    page: Any,
    event_id: str,
    screenshots: ScreenshotStore,
    snapshots: DomSnapshotter,
) -> Dict[str, Optional[str]]:
    return {
        "screenshot_path": screenshots.capture(page, event_id),
        "dom_snapshot_path": snapshots.capture_dom(page, event_id),
        "accessibility_tree_path": snapshots.capture_accessibility_tree(page, event_id),
    }


def append_events(
    logger: EventLogger,
    page: Any,
    raw_events: Iterable[Mapping[str, Any]],
    screenshots: ScreenshotStore,
    snapshots: DomSnapshotter,
) -> int:
    count = 0
    for raw_event in raw_events:
        event_id = logger.next_event_id()
        event = dict(raw_event)
        event["element"] = compact_element_info(event.get("element"))
        logger.append(
            event,
            event_id=event_id,
            page_context=page_context(page),
            artifacts=artifact_paths(page, event_id, screenshots, snapshots),
        )
        count += 1
    if count:
        logger.save()
    return count


def wait_for_enter(stop_event: threading.Event) -> None:
    try:
        input("Recording. Press Enter here to stop...\n")
    except EOFError:
        pass
    stop_event.set()


def run_recorder(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = EventLogger(
        output_dir,
        metadata={
            "start_url": args.url,
            "browser": args.browser,
            "headless": args.headless,
            "poll_interval_seconds": args.poll_interval,
        },
    )
    screenshots = ScreenshotStore(output_dir)
    snapshots = DomSnapshotter(output_dir)
    tracing_path = output_dir / "playwright_trace.zip"

    sync_playwright = load_playwright()
    stop_event = threading.Event()
    stop_thread = threading.Thread(target=wait_for_enter, args=(stop_event,), daemon=True)

    with sync_playwright() as playwright:
        browser_type = getattr(playwright, args.browser)
        browser = browser_type.launch(headless=args.headless)
        context = browser.new_context(viewport={"width": args.width, "height": args.height})
        context.add_init_script(RECORDER_SCRIPT)
        context.tracing.start(screenshots=True, snapshots=True, sources=True)
        page = context.new_page()

        page.goto(args.url, wait_until="domcontentloaded")
        install_recorder_script(page)
        append_events(
            logger,
            page,
            [{"action_type": "navigation", "url": page.url, "page_title": page.title()}],
            screenshots,
            snapshots,
        )

        last_url = page.url
        stop_thread.start()
        try:
            while not stop_event.is_set():
                install_recorder_script(page)
                raw_events = drain_browser_events(page)
                if page.url != last_url:
                    raw_events.append(
                        {"action_type": "navigation", "url": page.url, "page_title": page.title()}
                    )
                    last_url = page.url
                append_events(logger, page, raw_events, screenshots, snapshots)
                time.sleep(args.poll_interval)
        except KeyboardInterrupt:
            stop_event.set()
        finally:
            append_events(logger, page, drain_browser_events(page), screenshots, snapshots)
            context.tracing.stop(path=str(tracing_path))
            browser.close()

    logger.metadata["playwright_trace_path"] = tracing_path.name
    trace_path = logger.save()
    return trace_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record a browser demo as a Demo2Skill trace.")
    parser.add_argument("url", help="URL where the human demonstration should start.")
    parser.add_argument(
        "--output",
        "-o",
        default="runs/demo_trace",
        help="Directory for trace.json, screenshots, DOM snapshots, and Playwright trace.",
    )
    parser.add_argument(
        "--browser",
        choices=("chromium", "firefox", "webkit"),
        default="chromium",
        help="Playwright browser engine to use.",
    )
    parser.add_argument("--headless", action="store_true", help="Run without showing the browser.")
    parser.add_argument("--width", type=int, default=1280, help="Browser viewport width.")
    parser.add_argument("--height", type=int, default=900, help="Browser viewport height.")
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.35,
        help="Seconds between browser event drain attempts.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    trace_path = run_recorder(args)
    print(f"Saved trace to {trace_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

