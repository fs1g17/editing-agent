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
WORD_EDGE_TOL = 0.1  # seconds; Whisper word timings are roughly this precise


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


def word_context(t: float, words: list[dict]) -> dict:
    """Where does a cut at time t fall relative to the speech?"""
    if not words:
        return {}
    for idx, w in enumerate(words):
        if w["start"] <= t < w["end"]:
            # Inside a word. Whisper's word edges are only good to ~0.1 s, so cuts that close
            # to an edge are really on that edge.
            into = t - w["start"]
            if into <= WORD_EDGE_TOL:
                position = "at_word_start"
            elif w["end"] - t <= WORD_EDGE_TOL:
                position = "at_word_end"
            else:
                position = "mid_word"
            return {
                "position": position,
                "word": w["display"],
                "into_word": round(into, 3),
                "word_duration": round(w["end"] - w["start"], 3),
                "prev_word": words[idx - 1]["display"] if idx else None,
            }
        if t < w["start"]:
            prev = words[idx - 1] if idx else None
            gap_start = prev["end"] if prev else 0.0
            return {
                "position": "between_words",
                "prev_word": prev["display"] if prev else None,
                "next_word": w["display"],
                "after_prev_end": round(t - gap_start, 3),
                "before_next_start": round(w["start"] - t, 3),
                "gap": round(w["start"] - gap_start, 3),
                "after_sentence_end": bool(prev and prev["sentence_end"]),
                "after_clause_end": bool(prev and prev["clause_end"]),
            }
    return {"position": "after_speech", "prev_word": words[-1]["display"]}


def near_boundary(t: float, words: list[dict], tol: float = 0.12) -> str | None:
    """Is t within tol seconds of the end of a sentence / clause / any word?"""
    best = None
    for w in words:
        if abs(w["end"] - t) <= tol or (w["end"] <= t <= w["end"] + tol):
            level = "sentence" if w["sentence_end"] else "clause" if w["clause_end"] else "word"
            rank = {"sentence": 0, "clause": 1, "word": 2}
            if best is None or rank[level] < rank[best]:
                best = level
    return best
