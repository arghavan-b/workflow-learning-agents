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
                   choices=["auto", "screenvlm", "anthropic", "transformers", "scripted"],
                   help="Which parser backend to use. 'screenvlm' = the real "
                        "docling-project/ScreenVLM checkpoint (ScreenTag).")
    p.add_argument("--model", help="Model id: docling-project/ScreenVLM for --client "
                                   "screenvlm, or an HF VLM for --client transformers.")
    p.add_argument("--revision", help="Model revision/branch (e.g. v1 or v2 for ScreenVLM).")
    p.add_argument("--device", choices=["cpu", "mps", "cuda"],
                   help="Force compute device (default: auto — mps on Apple Silicon).")
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"],
                   help="Override model dtype (try float32 if mps/float16 output looks broken).")
    p.add_argument("--max-new-tokens", type=int,
                   help="Cap ScreenVLM output length (lower = much faster; default 6192).")
    p.add_argument("--image-max-edge", type=int, default=1024,
                   help="Downscale frames so the longest edge <= this before parsing "
                        "(big speedup on Retina screenshots; 0 disables). Default 1024.")
    p.add_argument("--ocr", default="none", choices=["none", "tesseract", "easyocr"],
                   help="OCR typed field values ScreenVLM omits. 'easyocr' is "
                        "pip-only (no system binary); 'tesseract' needs the engine.")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose (DEBUG) logging.")
    p.add_argument("--fps", type=float, default=1.0, help="Frame sampling rate (sample=fps).")
    p.add_argument("--sample", default="fps", choices=["fps", "keyframes", "scene"],
                   help="Frame sampling strategy for video decode.")
    p.add_argument("--scene-threshold", type=float, default=0.3,
                   help="Scene-cut sensitivity (0..1) for --sample scene.")
    p.add_argument("--max-frames", type=int, help="Cap frames parsed (each is one model call).")
    p.add_argument("--raw-out", help="Dir to dump ScreenVLM's raw ScreenTag per frame (debug).")
    p.add_argument("--cursor", help="A cursor records JSON to aid click disambiguation.")
    p.add_argument("-o", "--states-out", required=True, help="Where to write parsed states JSON.")
    p.add_argument("--trace-out", help="Also derive actions and write a normalize-ready raw trace.")
    p.add_argument("--graph", action="store_true", help="Print the induced UI state graph.")
    p.add_argument("--video-id", default="parsed_video")
    return p


def _client(args):
    if args.client == "anthropic":
        from demo2skill.video.statediff.parser.clients import AnthropicVisionClient
        return AnthropicVisionClient()
    if args.client == "transformers":
        from demo2skill.video.statediff.parser.clients import TransformersScreenVLMClient
        import os
        model = args.model or os.environ.get("SCREENVLM_MODEL")
        return TransformersScreenVLMClient(model) if model else TransformersScreenVLMClient()
    from demo2skill.video.statediff.parser.clients import default_screen_parser_client
    return default_screen_parser_client()


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

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
                                     ocr=make_ocr(args.ocr))
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

    # -- parse ----------------------------------------------------------------
    log.info("extracted %d frame(s); parsing with '%s'", len(frames), args.client)
    states = parse_frames(parser, frames)

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
