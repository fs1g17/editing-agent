"""Shared paths and helpers for the style-analyser scripts."""

from __future__ import annotations

import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REFERENCE_DIR = ROOT / "reference"
ANALYSIS_DIR = ROOT / "analysis"


def analysis_dir_for(video: Path) -> Path:
    """reference/<creator>/<name>.mp4 -> analysis/<creator>/<name>/"""
    try:
        rel = video.resolve().relative_to(REFERENCE_DIR)
        return ANALYSIS_DIR / rel.parent / video.stem
    except ValueError:  # a video outside reference/
        return ANALYSIS_DIR / "_other" / video.stem


def rel(path: Path) -> str:
    path = path.resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def collect_videos(paths: list[str]) -> list[Path]:
    targets = [Path(p) for p in paths] or [REFERENCE_DIR]
    videos: list[Path] = []
    for t in targets:
        if t.is_dir():
            videos += sorted(t.rglob("*.mp4"))
        elif t.is_file():
            videos.append(t)
        else:
            print(f"  !! not found: {t}", file=sys.stderr)
    return videos


def probe_video(video: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate,nb_frames:format=duration",
         "-of", "json", str(video)],
        check=True, capture_output=True, text=True,
    ).stdout
    data = json.loads(out)
    stream = data["streams"][0]
    fps = Fraction(stream["r_frame_rate"])
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": float(fps),
        "fps_fraction": f"{fps.numerator}/{fps.denominator}",
        "duration": float(data["format"]["duration"]),
    }


def decode_frames(video: Path, width: int, height: int, gray: bool = False) -> np.ndarray:
    """Decode every frame, resized to width x height. Returns uint8 [N, H, W, 3] (or [N, H, W])."""
    pix_fmt, channels = ("gray", 1) if gray else ("rgb24", 3)
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-f", "rawvideo", "-pix_fmt", pix_fmt,
         "-s", f"{width}x{height}", "pipe:"],
        check=True, capture_output=True,
    ).stdout
    shape = (-1, height, width) if gray else (-1, height, width, channels)
    return np.frombuffer(raw, np.uint8).reshape(shape)


def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
