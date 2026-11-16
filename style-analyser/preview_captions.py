"""Render a caption-detection preview to compare against the real video.

Usage:
    uv run preview_captions.py reference/kallawaymarketing/2026-11-08_DffZRmfMWrf.mp4

Writes to analysis/<creator>/<name>/captions/:
    preview.mp4    the original video (with audio) plus:
                     - a green box around every detected caption line
                     - a bar underneath: detected caption text, its timing/lead, size and colour,
                       and the word Whisper hears at that moment (so mismatches stand out)
    captions.srt   the detected captions as subtitles (load next to the original in any player)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from common import analysis_dir_for, load_json, probe_video, rel
from detect_faces import iter_frames

BAR_H = 260


def srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(captions: list[dict], path: Path) -> None:
    lines = []
    for i, c in enumerate(captions, 1):
        lines += [str(i), f"{srt_time(c['start'])} --> {srt_time(c['end'])}", c["text"], ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def put(img, text, org, scale=1.0, color=(255, 255, 255), thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def ascii_only(s: str) -> str:
    """OpenCV can only draw ASCII: swap typographic quotes/dashes for plain ones first."""
    s = s.translate(str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'",
                                   "–": "-", "—": "-", "…": "..."}))
    return s.encode("ascii", "replace").decode()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", help="Reference video (captions.json must exist)")
    args = parser.parse_args()

    video = Path(args.video)
    out_dir = analysis_dir_for(video)
    data = load_json(out_dir / "captions.json")
    if not data:
        parser.error("no captions.json - run detect_captions.py first")
    transcript = load_json(out_dir / "transcript.json") or {"words": []}
    words = transcript["words"]
    caps = data["captions"]
    info = probe_video(video)
    W, H, fps = info["width"], info["height"], info["fps"]

    cap_dir = out_dir / "captions"
    cap_dir.mkdir(parents=True, exist_ok=True)
    write_srt(caps, cap_dir / "captions.srt")

    out_path = cap_dir / "preview.mp4"
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H + BAR_H}", "-r", info["fps_fraction"], "-i", "pipe:",
         "-i", str(video), "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out_path)],
        stdin=subprocess.PIPE,
    )
    by_frame: dict[int, dict] = {}
    for c in caps:
        for f in range(c["start_frame"], c["end_frame"] + 1):
            by_frame[f] = c
    wi = 0
    for f, frame in enumerate(iter_frames(video, W, H)):
        t = f / fps
        canvas = np.zeros((H + BAR_H, W, 3), np.uint8)
        canvas[:H] = frame
        c = by_frame.get(f)
        if c:
            for line in c["lines"]:
                x0, y0, x1, y1 = line["box"]
                cv2.rectangle(canvas, (int(x0 * W) - 6, int(y0 * H) - 6), (int(x1 * W) + 6, int(y1 * H) + 6),
                              (80, 255, 80) if not c["emphasis"] else (0, 200, 255), 3)
        # current spoken word(s)
        while wi < len(words) and words[wi]["end"] < t:
            wi += 1
        spoken = [w["display"] for w in words[max(0, wi - 1):wi + 2] if w["start"] - 0.05 <= t <= w["end"] + 0.05]
        y = H + 50
        put(canvas, f"{t:6.2f}s  frame {f}", (20, y), 1.1, (180, 180, 180))
        if c:
            col = c.get("colour", {})
            put(canvas, f"OCR: {ascii_only(c['text'])}", (20, y + 60), 1.5, (80, 255, 80))
            put(canvas, f"{c['start']:.2f}-{c['end']:.2f}s  lead {c['lead_s']}  h {c['lines'][0]['height']:.3f}  "
                        f"{col.get('fill')}  {'EMPH ' if c['emphasis'] else ''}{c['entry_animation']}",
                (20, y + 110), 0.9, (220, 220, 220))
        else:
            put(canvas, "OCR: -", (20, y + 60), 1.5, (120, 120, 120))
        put(canvas, f"SAID: {ascii_only(' '.join(spoken)) or '-'}", (20, y + 170), 1.5, (90, 200, 255))
        enc.stdin.write(canvas.tobytes())
    enc.stdin.close()
    enc.wait()
    print(f"-> {rel(out_path)}\n-> {rel(cap_dir / 'captions.srt')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
