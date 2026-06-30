"""Frame access for the inverse-dynamics module.

A :class:`Frames` is the observation stream the detector/recognizer read. It can
be built from a directory of pre-extracted frames or decoded from a video with
ffmpeg (best-effort; absence of ffmpeg is not fatal - the scripted backend needs
no pixels). Windowing helpers let the recognizer look only at the frames around
a detected action interval.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


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
    def from_dir(cls, directory, fps: float = 1.0) -> "Frames":
        directory = Path(directory)
        paths = sorted(p for p in directory.glob("*.png")) or sorted(directory.glob("*.jpg"))
        step_ms = int(1000 / fps) if fps else 1000
        frames = [Frame(index=i, ms=i * step_ms, path=p) for i, p in enumerate(paths)]
        return cls(frames, fps=fps)

    @classmethod
    def from_video(cls, video_path, fps: float = 2.0, out_dir=None) -> "Frames":
        """Decode ``video_path`` to frames at ``fps`` using ffmpeg if available."""

        video_path = Path(video_path)
        out_dir = Path(out_dir) if out_dir else video_path.with_suffix("") / "frames"
        out_dir.mkdir(parents=True, exist_ok=True)
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "ffmpeg not found; pre-extract frames and use Frames.from_dir, "
                "or use the scripted backend which needs no pixels."
            )
        subprocess.run(
            ["ffmpeg", "-i", str(video_path), "-vf", f"fps={fps}",
             str(out_dir / "frame_%06d.png")],
            check=True, capture_output=True,
        )
        return cls.from_dir(out_dir, fps=fps)

    @classmethod
    def empty(cls) -> "Frames":
        """A pixel-free stream (the scripted backend only needs timestamps)."""

        return cls([], fps=1.0)
