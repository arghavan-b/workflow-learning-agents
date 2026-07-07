"""``demo2skill-parse-video``: run the pixels->ScreenState front over a video.

This exercises the parsing part of the state-diff engine in isolation: it turns
frames into parsed screen states (and, optionally, the action trajectory and the
UI state graph the IDM derives from them).

Three input modes:

* ``--replay states.json`` - no model, no cost: replay pre-parsed states through
  the same pipeline. Use this first to check the wiring end to end.
* a video file - decode frames with ffmpeg, then parse each with a VLM.
* ``--frames-dir DIR`` - parse pre-extracted frames (skips ffmpeg).

Parser client is chosen with ``--client``:
  auto (env-driven) | anthropic | transformers | scripted (with --replay).

Examples
--------
    # 1. Free dry run against the bundled example (no API, no ffmpeg):
    demo2skill-parse-video \\
        --replay demo2skill/examples/github_issue/screen_states.json \\
        -o runs/parse/states.json --trace-out runs/parse/trace.json --graph

    # 2. A real screen recording via Claude vision (needs ANTHROPIC_API_KEY):
    demo2skill-parse-video demo.mp4 --client anthropic --fps 1 --max-frames 40 \\
        -o runs/parse/states.json --trace-out runs/parse/trace.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

from demo2skill.video.statediff import (
    ScriptedScreenParser,
    StateTrajectoryBuilder,
    UIStateGraph,
    VLMScreenParser,
    load_states,
    parse_frames,
    states_payload,
)
from demo2skill.video.statediff.cursor import CursorTrack
from demo2skill.video.video2action.frames import Frame, Frames


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Parse a video's frames into screen states.")
    p.add_argument("video", nargs="?", help="Path to a screen-recording video (mp4, ...).")
    p.add_argument("--frames-dir", help="Directory of pre-extracted frames (skips ffmpeg).")
    p.add_argument("--replay", help="A screen_states.json to replay (no model, no cost).")
    p.add_argument("--client", default="auto",
                   choices=["auto", "screenvlm", "anthropic", "openai", "transformers", "scripted"],
                   help="Which parser backend to use. 'screenvlm' = the real "
                        "docling-project/ScreenVLM checkpoint (ScreenTag); "
                        "'anthropic'/'openai' = a general vision model (JSON).")
    p.add_argument("--model", help="Model id: docling-project/ScreenVLM for --client "
                                   "screenvlm, or an HF VLM for --client transformers.")
    p.add_argument("--revision", help="Model revision/branch (e.g. v1 or v2 for ScreenVLM).")
    p.add_argument("--device", choices=["cpu", "mps", "cuda"],
                   help="Force compute device (default: auto — mps on Apple Silicon).")
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"],
                   help="Override model dtype (try float32 if mps/float16 output looks broken).")
    p.add_argument("--max-new-tokens", type=int,
                   help="Cap ScreenVLM output length (lower = much faster; default 6192).")
    p.add_argument("--invert", action="store_true",
                   help="Invert frame colors (dark->light) before parsing — helps "
                        "light-trained parsers/OCR on dark-mode UIs. Cursor "
                        "detection still uses the original frames.")
    p.add_argument("--image-max-edge", type=int, default=1024,
                   help="Downscale frames so the longest edge <= this before parsing "
                        "(big speedup on Retina screenshots; 0 disables). Default 1024.")
    p.add_argument("--ocr", default="none",
                   choices=["none", "tesseract", "easyocr", "paddle"],
                   help="OCR typed field values ScreenVLM omits. 'easyocr' and "
                        "'paddle' are pip-only (no system binary); 'tesseract' "
                        "needs the engine.")
    p.add_argument("--ocr-version",
                   help="PaddleOCR model version, e.g. PP-OCRv6 or PP-OCRv5 "
                        "(needs paddleocr>=3.7 for v6). Default: library default.")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose (DEBUG) logging.")
    p.add_argument("--fps", type=float, default=1.0, help="Frame sampling rate (sample=fps).")
    p.add_argument("--sample", default="fps", choices=["fps", "keyframes", "scene"],
                   help="Frame sampling strategy for video decode.")
    p.add_argument("--scene-threshold", type=float, default=0.3,
                   help="Scene-cut sensitivity (0..1) for --sample scene.")
    p.add_argument("--max-frames", type=int, help="Cap frames parsed (each is one model call).")
    p.add_argument("--raw-out", help="Dir to dump ScreenVLM's raw ScreenTag per frame (debug).")
    p.add_argument("--cursor", help="A cursor records JSON to aid click disambiguation.")
    p.add_argument("--detect-cursor", default="none", choices=["none", "template"],
                   help="Recover cursor positions from the frames themselves "
                        "(needs the 'cursor' extra). Ignored if --cursor is given.")
    p.add_argument("--cursor-template",
                   help="A crop of your OS cursor for --detect-cursor template "
                        "(a real crop beats the synthetic fallback).")
    p.add_argument("-o", "--states-out", required=True, help="Where to write parsed states JSON.")
    p.add_argument("--trace-out", help="Also derive actions and write a normalize-ready raw trace.")
    p.add_argument("--graph", action="store_true", help="Print the induced UI state graph.")
    p.add_argument("--video-id", default="parsed_video")
    return p


def _client(args):
    if args.client == "anthropic":
        from demo2skill.video.statediff.parser.clients import AnthropicVisionClient
        return AnthropicVisionClient(args.model) if args.model else AnthropicVisionClient()
    if args.client == "openai":
        from demo2skill.video.statediff.parser.clients import OpenAIVisionClient
        return OpenAIVisionClient(args.model) if args.model else OpenAIVisionClient()
    if args.client == "transformers":
        from demo2skill.video.statediff.parser.clients import TransformersScreenVLMClient
        import os
        model = args.model or os.environ.get("SCREENVLM_MODEL")
        return TransformersScreenVLMClient(model) if model else TransformersScreenVLMClient()
    from demo2skill.video.statediff.parser.clients import default_screen_parser_client
    return default_screen_parser_client()


def _load_dotenv() -> None:
    """Load KEY=VALUE lines from a local ``.env`` into the environment (without
    overriding anything already set). No dependency; keeps API keys out of the
    command line."""
    import os
    for path in (Path.cwd() / ".env", Path(__file__).resolve().parents[3] / ".env"):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _load_dotenv()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
    )
    log = logging.getLogger("demo2skill.parse-video")

    cursor = CursorTrack.empty()
    if args.cursor:
        cursor = CursorTrack.from_records(json.loads(Path(args.cursor).read_text()))

    # -- pick parser + frames -------------------------------------------------
    if args.replay or args.client == "scripted":
        if not args.replay:
            print("--client scripted requires --replay states.json", file=sys.stderr)
            return 2
        data = json.loads(Path(args.replay).read_text())
        replay_states, replay_cursor = load_states(data)
        if not args.cursor:
            cursor = replay_cursor
        parser = ScriptedScreenParser(data["states"])
        frames = Frames([Frame(index=s.index, ms=s.ms) for s in replay_states])
    else:
        if args.client == "screenvlm":
            from demo2skill.video.statediff import ScreenVLMParser
            from demo2skill.video.statediff.parser.ocr import make_ocr
            parser = ScreenVLMParser(args.model or "docling-project/ScreenVLM",
                                     revision=args.revision, raw_dir=args.raw_out,
                                     device=args.device, dtype=args.dtype,
                                     max_new_tokens=args.max_new_tokens or 6192,
                                     max_image_edge=args.image_max_edge,
                                     ocr=make_ocr(args.ocr, ocr_version=args.ocr_version))
        else:
            client = _client(args)
            if client is None:
                print("No parser client configured. Set ANTHROPIC_API_KEY or SCREENVLM_MODEL, "
                      "pass --client screenvlm/anthropic/transformers, or --replay for a dry run.",
                      file=sys.stderr)
                return 2
            parser = VLMScreenParser(client)
        if args.frames_dir:
            frames = Frames.from_dir(args.frames_dir, fps=args.fps)
        elif args.video:
            frames = Frames.from_video(
                args.video, fps=args.fps, sample=args.sample,
                scene_threshold=args.scene_threshold, max_frames=args.max_frames,
            )
        else:
            print("Provide a video path, --frames-dir, or --replay.", file=sys.stderr)
            return 2

    if args.max_frames:
        frames = Frames(frames.frames[: args.max_frames], fps=frames.fps,
                        width=frames.width, height=frames.height)

    # -- recover cursor from frames (optional) --------------------------------
    if args.detect_cursor != "none" and not args.cursor:
        from demo2skill.video.statediff.cursor_detect import (
            build_cursor_detector, detect_cursor_track,
        )
        det = build_cursor_detector(args.detect_cursor, template_path=args.cursor_template)
        log.info("detecting cursor over %d frame(s)...", len(frames))
        cursor = detect_cursor_track(det, frames)

    # -- parse ----------------------------------------------------------------
    log.info("extracted %d frame(s); parsing with '%s'%s", len(frames), args.client,
             " (inverted)" if args.invert else "")
    states = parse_frames(parser, frames, invert=args.invert)

    # Parse-stability gate: report count swing, then drop transient (flicker)
    # elements so they don't flood the IDM with phantom transitions.
    counts = [len(s.elements) for s in states]
    if counts:
        import statistics
        med = statistics.median(counts) or 1
        dev = max(abs(c - med) for c in counts)
        log.info("parse stability: elements min=%d max=%d median=%.0f (max deviation "
                 "%.0f%%)%s", min(counts), max(counts), med, 100 * dev / med,
                 "  ⚠ unstable" if dev > 0.5 * med else "")
    from demo2skill.video.statediff.stability import stabilize_states
    stabilize_states(states)

    # Temporal placeholder-vs-value classification (Option 1): reclassify field
    # text across frames so placeholders don't masquerade as typed values.
    from demo2skill.video.statediff.field_text import classify_field_text
    classify_field_text(states)

    out = Path(args.states_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(states_payload(states, cursor), indent=2, ensure_ascii=False),
                   encoding="utf-8")

    traj = None
    if args.trace_out:
        traj = StateTrajectoryBuilder(cursor).build(states, video_id=args.video_id)
        trace_path = Path(args.trace_out)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(json.dumps(traj.to_raw_trace(), indent=2, ensure_ascii=False),
                              encoding="utf-8")

    graph = UIStateGraph.build(states, cursor)

    print(f"frames parsed: {len(frames)}")
    print(f"screen states: {len(states)}  ->  distinct nodes: {len(graph.nodes)}")
    if traj is not None:
        kinds = [a.action_type for a in traj.actions]
        print(f"actions:       {len(traj.actions)}  {kinds}")
        print(f"  states -> {out}")
        print(f"  trace  -> {args.trace_out}")
    else:
        print(f"  states -> {out}")
    if args.graph:
        print("\nUI state graph:")
        print(graph.ascii())
    return 0


if __name__ == "__main__":
    sys.exit(main())
