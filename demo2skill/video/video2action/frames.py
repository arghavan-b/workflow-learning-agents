"""Frame access for the inverse-dynamics module.

A :class:`Frames` is the observation stream the detector / recognizer / screen
parser read. It can be built from a directory of pre-extracted frames or decoded
from a video with ffmpeg. Decoding supports three sampling strategies so you can
trade coverage against model cost:

* ``fps``       - uniform sampling at N frames/second (default);
* ``keyframes`` - only encoded I-frames (cheap; great for slide-like tutorials);
* ``scene``     - frames at scene-change boundaries above a threshold (adaptive;
                  good for busy UIs with animation/video content).

ffmpeg reads essentially any container (mp4, mov, mkv, webm, avi, ...), so format
is a non-issue. Real per-frame timestamps are recovered from the filter's
``metadata=print`` output rather than faked, and resolution / fps are probed with
ffprobe when available. Absence of ffmpeg is not fatal for the scripted / replay
paths - only ``from_video`` needs it.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger("demo2skill.parser")

_PTS_RE = re.compile(r"pts_time:([0-9]+(?:\.[0-9]+)?)")
SAMPLE_MODES = ("fps", "keyframes", "scene")


@dataclass
class Frame:
    index: int
    ms: int
    path: Optional[Path] = None  # None when frames live only as timestamps

    def bytes(self) -> Optional[bytes]:
        return self.path.read_bytes() if self.path and self.path.exists() else None


class Frames:
    def __init__(self, frames: List[Frame], fps: float = 1.0,
                 width: Optional[int] = None, height: Optional[int] = None) -> None:
        self.frames = sorted(frames, key=lambda f: f.ms)
        self.fps = fps
        self.width = width
        self.height = height

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def duration_ms(self) -> int:
        return self.frames[-1].ms if self.frames else 0

    def at(self, ms: int) -> Optional[Frame]:
        if not self.frames:
            return None
        return min(self.frames, key=lambda f: abs(f.ms - ms))

    def window(self, start_ms: int, end_ms: int) -> List[Frame]:
        return [f for f in self.frames if start_ms <= f.ms <= end_ms]

    # -- constructors --------------------------------------------------------

    @classmethod
    def from_dir(cls, directory, fps: float = 1.0,
                 width: Optional[int] = None, height: Optional[int] = None) -> "Frames":
        directory = Path(directory)
        exts = ("*.png", "*.jpg", "*.jpeg", "*.webp")
        paths: List[Path] = []
        for ext in exts:
            paths.extend(directory.glob(ext))
        paths = sorted(paths)
        step_ms = int(1000 / fps) if fps else 1000
        frames = [Frame(index=i, ms=i * step_ms, path=p) for i, p in enumerate(paths)]
        return cls(frames, fps=fps, width=width, height=height)

    @classmethod
    def from_video(
        cls,
        video_path,
        fps: float = 2.0,
        out_dir=None,
        *,
        sample: str = "fps",
        scene_threshold: float = 0.3,
        max_frames: Optional[int] = None,
    ) -> "Frames":
        """Decode ``video_path`` to frames with ffmpeg.

        ``sample`` selects the strategy (``fps`` | ``keyframes`` | ``scene``).
        ``fps`` is used only for uniform sampling; ``scene_threshold`` (0..1) sets
        the scene-cut sensitivity for ``sample='scene'``. Per-frame timestamps are
        read back from ffmpeg rather than assumed.
        """

        if sample not in SAMPLE_MODES:
            raise ValueError(f"sample must be one of {SAMPLE_MODES}, got {sample!r}")

        video_path = Path(video_path)
        out_dir = Path(out_dir) if out_dir else video_path.with_suffix("") / "frames"
        out_dir.mkdir(parents=True, exist_ok=True)
        exe = _ffmpeg_exe()
        if exe is None:
            raise RuntimeError(
                "ffmpeg not found. Install it one of these ways:\n"
                "  pip-only (no brew):  uv pip install imageio-ffmpeg\n"
                "  system:              brew install ffmpeg   (macOS)\n"
                "Or pre-extract frames yourself and use Frames.from_dir / --frames-dir."
            )

        width, height, probed_fps, _duration = probe_video(video_path)
        times_file = out_dir / "frame_times.txt"
        out_pattern = out_dir / "frame_%06d.png"
        select = _select_filter(sample, fps=fps, scene_threshold=scene_threshold)

        # First try with per-frame timestamp printing; if that fails (some ffmpeg
        # builds/paths reject the metadata filter), retry the plain filter and
        # fall back to synthetic timestamps rather than erroring out.
        vf_meta = f"{select},metadata=print:file={times_file}"
        proc = _run_ffmpeg(exe, video_path, vf_meta, out_pattern, max_frames)
        used_meta = proc.returncode == 0
        if not used_meta:
            logger.warning("ffmpeg timing pass failed (exit %s); retrying without "
                           "metadata (synthetic timestamps)", proc.returncode)
            for p in out_dir.glob("frame_*.png"):
                p.unlink()
            proc = _run_ffmpeg(exe, video_path, select, out_pattern, max_frames)
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed to decode {video_path} (exit {proc.returncode}).\n"
                + ((proc.stderr or "").strip()[-1000:] or "(no stderr)")
            )

        paths = sorted(out_dir.glob("frame_*.png"))
        times_ms = _read_pts_ms(times_file) if used_meta else []
        step_ms = int(1000 / fps) if fps else 1000
        frames: List[Frame] = []
        for i, p in enumerate(paths):
            ms = times_ms[i] if i < len(times_ms) else i * step_ms
            frames.append(Frame(index=i, ms=ms, path=p))
        return cls(frames, fps=(probed_fps or fps), width=width, height=height)

    @classmethod
    def empty(cls) -> "Frames":
        """A pixel-free stream (the scripted backend only needs timestamps)."""

        return cls([], fps=1.0)


# -- ffmpeg / ffprobe helpers (pure parsing split out for testability) --------

def _ffmpeg_exe() -> Optional[str]:
    """Resolve an ffmpeg binary: PATH first, then the pip-installable
    ``imageio-ffmpeg`` bundle (no system install / brew needed)."""

    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _select_filter(sample: str, *, fps: float, scene_threshold: float) -> str:
    if sample == "keyframes":
        return "select='eq(pict_type\\,I)'"
    if sample == "scene":
        return f"select='gt(scene\\,{scene_threshold})'"
    return f"fps={fps}"  # uniform


def _filter_for(sample: str, *, fps: float, scene_threshold: float, times_file: Path) -> str:
    # metadata=print emits one block per output frame (in order) carrying pts_time.
    return (f"{_select_filter(sample, fps=fps, scene_threshold=scene_threshold)}"
            f",metadata=print:file={times_file}")


def _run_ffmpeg(exe: str, video_path, vf: str, out_pattern, max_frames):
    cmd = [exe, "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
           "-i", str(video_path), "-vf", vf, "-vsync", "vfr"]
    if max_frames:
        cmd += ["-frames:v", str(int(max_frames))]
    cmd += [str(out_pattern)]
    return subprocess.run(cmd, capture_output=True, text=True)


def _read_pts_ms(times_file: Path) -> List[int]:
    if not times_file.exists():
        return []
    return _parse_pts_ms(times_file.read_text(encoding="utf-8", errors="ignore"))


def _parse_pts_ms(text: str) -> List[int]:
    """Extract per-frame timestamps (ms) from ffmpeg ``metadata=print`` output."""

    return [int(round(float(m) * 1000)) for m in _PTS_RE.findall(text)]


def probe_video(video_path) -> Tuple[Optional[int], Optional[int], Optional[float], Optional[float]]:
    """Return ``(width, height, fps, duration_s)`` via ffprobe, best-effort."""

    if shutil.which("ffprobe") is None:
        return (None, None, None, None)
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,r_frame_rate:format=duration",
             "-of", "default=noprint_wrappers=1", str(video_path)],
            check=True, capture_output=True, text=True,
        ).stdout
    except (subprocess.CalledProcessError, OSError):
        return (None, None, None, None)
    return _parse_probe(out)


def _parse_probe(text: str):
    fields = {}
    for line in text.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            fields[k.strip()] = v.strip()

    def _int(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    def _rate(v):
        if not v or v in ("0/0", "N/A"):
            return None
        if "/" in v:
            num, _, den = v.partition("/")
            try:
                d = float(den)
                return float(num) / d if d else None
            except ValueError:
                return None
        try:
            return float(v)
        except ValueError:
            return None

    width = _int(fields.get("width"))
    height = _int(fields.get("height"))
    fps = _rate(fields.get("r_frame_rate"))
    try:
        duration = float(fields.get("duration")) if fields.get("duration") not in (None, "N/A") else None
    except ValueError:
        duration = None
    return (width, height, fps, duration)
