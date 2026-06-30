"""``demo2skill-video2action``: raw tutorial video -> trajectory (-> skill).

    # deterministic, no model (overlay/sidecar-driven):
    demo2skill-video2action --events demo_events.json -o runs/issue_video --induce

    # from extracted frames with a real VLM backend (wire your own VLMClient):
    demo2skill-video2action --frames runs/issue_video/frames --fps 2 -o runs/issue_video

Writes ``trajectory.json`` and a normalize-ready ``trace.json``. With
``--induce`` it also runs the existing normalize + induction pipeline and writes
``semantic_trace.json`` and ``workflow.yaml`` - the full raw-video-to-skill path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from demo2skill.video.video2action.backends.scripted import ScriptedBackend
from demo2skill.video.video2action.frames import Frames
from demo2skill.video.video2action.idm import Video2Action


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Induce a trajectory from a tutorial video.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="Video file (decoded via ffmpeg).")
    src.add_argument("--frames", help="Directory of pre-extracted frames.")
    src.add_argument("--events", help="Sidecar JSON of action records (scripted backend).")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--video-id", default=None)
    parser.add_argument("-o", "--output", required=True, help="Output directory.")
    parser.add_argument("--induce", action="store_true",
                        help="Also normalize + induce a workflow skill.")
    return parser


def _build_backend(args):
    """Pick a backend. Scripted when --events is given; otherwise a VLM backend
    is required (left to the caller to wire, since no model is bundled)."""

    if args.events:
        records = json.loads(Path(args.events).read_text(encoding="utf-8"))
        backend = ScriptedBackend(records)
        return backend, backend, Frames.empty()
    raise SystemExit(
        "Pixel input given but no VLM backend is wired in this build. Provide "
        "--events for the deterministic path, or construct Video2Action with a "
        "VLMClient (see demo2skill/video/video2action/backends/vlm.py)."
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.output)
    video_id = args.video_id or out_dir.name

    detector, recognizer, frames = _build_backend(args)
    source = args.events or args.frames or args.video

    trajectory = Video2Action(detector, recognizer).run(
        frames, video_id=video_id, source=source
    )

    _write(out_dir / "trajectory.json", trajectory.to_dict())
    raw_trace = trajectory.to_raw_trace()
    _write(out_dir / "trace.json", raw_trace)
    print(f"Wrote {len(trajectory.actions)} actions -> {out_dir/'trajectory.json'}")
    print(f"Wrote raw trace -> {out_dir/'trace.json'}")

    if args.induce:
        from demo2skill.trace.normalize import normalize_trace
        from demo2skill.induction.workflow_generator import induce_workflow

        semantic = normalize_trace(raw_trace)
        _write(out_dir / "semantic_trace.json", semantic)
        skill = induce_workflow(semantic)
        (out_dir / "workflow.yaml").write_text(skill.to_yaml(), encoding="utf-8")
        print(f"Induced skill '{skill.workflow_id}' -> {out_dir/'workflow.yaml'}")
        print(f"  inputs: {skill.input_names}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
