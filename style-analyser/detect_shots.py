"""Detect shots, cuts and transitions with TransNetV2, and line them up with the transcript.

Usage:
    uv run detect_shots.py                              # every video under reference/
    uv run detect_shots.py reference/kallawaymarketing  # one creator
    uv run detect_shots.py video.mp4 --force --threshold 0.4

Writes to analysis/<creator>/<name>/:
    shots.json          shots, transitions (type + length), jump-cut candidates, pacing stats,
                        and how each cut sits against the words (needs transcript.json for that part)
    shots_raw.npz       per-frame TransNetV2 probabilities + frame-difference/brightness curves,
                        so thresholds can be re-tuned without re-running the model
    strips/strip_NN.jpg frame strips around every boundary, for checking cuts by eye or with a VLM

Measurements only: transition `type` is a heuristic from the curves (cut / gradual / fade / flash);
telling a dissolve from a whip pan or a zoom transition is left for the visual pass.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from common import (analysis_dir_for, collect_videos, decode_frames, load_json, near_boundary, probe_video,
                    rel, save_json, word_context)

# Small frames used for difference/brightness curves and the strips (9:16).
SMALL_W, SMALL_H = 108, 192
STRIP_BEFORE, STRIP_AFTER = 4, 4      # frames shown either side of a boundary
STRIPS_PER_PAGE = 8
JUMP_MIN_GAP = 4                      # jump-cut candidates closer than this to a TransNetV2 cut are ignored


# ---------------------------------------------------------------- model

_model = None


def get_model():
    global _model
    if _model is None:
        from transnetv2_pytorch import TransNetV2  # slow import, only when needed
        _model = TransNetV2(device="auto")
        _model.eval()
    return _model


def run_transnet(video: Path) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame probabilities: (cut here, frame is part of a transition)."""
    model = get_model()
    frames = decode_frames(video, 48, 27)
    tensor = torch.from_numpy(frames.copy()).to(model.device)
    with torch.no_grad():
        single, many = model.predict_frames(tensor, quiet=True)
    return single.numpy().astype(np.float32), many.numpy().astype(np.float32)


# ---------------------------------------------------------------- signals

def frame_signals(small: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """diff[i] = mean |frame i - frame i-1| (0..1), luma[i] = mean brightness (0..1)."""
    gray = small.astype(np.float32).mean(axis=3) / 255.0
    diff = np.zeros(len(gray), np.float32)
    diff[1:] = np.abs(gray[1:] - gray[:-1]).mean(axis=(1, 2))
    luma = gray.mean(axis=(1, 2))
    return diff, luma


def find_boundaries(single: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Runs of frames with cut probability above threshold -> (first, last) frame of each run."""
    above = single > threshold
    runs, start = [], None
    for i, flag in enumerate(above):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(above) - 1))
    return runs


def transition_span(many: np.ndarray, peak: int, threshold: float = 0.5) -> tuple[int, int]:
    """Grow outwards from the peak while the 'inside a transition' probability stays high."""
    lo = hi = peak
    while lo - 1 >= 0 and many[lo - 1] > threshold:
        lo -= 1
    while hi + 1 < len(many) and many[hi + 1] > threshold:
        hi += 1
    return lo, hi


def classify(diff: np.ndarray, luma: np.ndarray, lo: int, hi: int) -> tuple[str, dict]:
    """Heuristic transition type from the curves around [lo, hi]."""
    a, b = max(1, lo - 3), min(len(diff), hi + 5)
    window = diff[a:b]
    total = float(window.sum()) or 1e-6
    concentration = float(window.max()) / total          # ~1.0 when all change happens in one frame
    span_frames = hi - lo + 1
    base_luma = float(np.median(np.r_[luma[max(0, a - 12):a], luma[b:b + 12]])) if b < len(luma) else float(luma[a])
    min_luma, max_luma = float(luma[a:b].min()), float(luma[a:b].max())

    # TransNetV2's transition span is the reliable signal: hard cuts are 1-2 frames even when
    # only part of the frame changes (split-screen) or the new shot starts black and animates in.
    span_luma = luma[lo:hi + 1]
    if span_frames <= 2:
        kind = "cut"
    elif span_luma.min() < 0.06 and base_luma > 0.12:
        kind = "fade_black"
    elif span_luma.max() > 0.85 and span_luma.max() - base_luma > 0.25:
        kind = "flash"
    else:
        kind = "gradual"
    return kind, {
        "diff_concentration": round(concentration, 3),
        "peak_diff": round(float(window.max()), 4),
        "luma_min": round(min_luma, 3),
        "luma_max": round(max_luma, 3),
    }


def jump_cut_candidates(diff: np.ndarray, cut_frames: list[int], fps: float) -> list[dict]:
    """Isolated one-frame spikes in the difference curve that TransNetV2 did not call a cut.
    In talking-head edits these are usually jump cuts (pause removed, same framing) or
    punch-ins. Candidates only: check them in the strips before trusting them."""
    out = []
    half = max(6, int(round(fps / 2)))
    for i in range(2, len(diff) - 2):
        if any(abs(i - c) < JUMP_MIN_GAP for c in cut_frames):
            continue
        neighbours = np.r_[diff[max(1, i - half):i - 1], diff[i + 2:i + half + 1]]
        if len(neighbours) < 4:
            continue
        base = float(np.median(neighbours))
        spike = float(diff[i])
        isolated = spike > 2.5 * max(diff[i - 1], diff[i + 1])
        if isolated and spike > max(4 * base, base + 0.025):
            out.append({"frame": i, "time": round(i / fps, 3), "diff": round(spike, 4),
                        "baseline": round(base, 4), "ratio": round(spike / max(base, 1e-4), 1)})
    return out


def changed_area(small: np.ndarray, frame: int, grid: int = 4, thresh: float = 0.06) -> dict:
    """Which parts of the frame changed between frame-1 and frame? Split-screen layouts often
    cut only the top or bottom half, which TransNetV2 tends to miss."""
    a = small[frame - 1].astype(np.float32).mean(axis=2) / 255.0
    b = small[frame].astype(np.float32).mean(axis=2) / 255.0
    h, w = a.shape
    cells = np.zeros((grid, grid), bool)
    for r in range(grid):
        for c in range(grid):
            ys, xs = slice(r * h // grid, (r + 1) * h // grid), slice(c * w // grid, (c + 1) * w // grid)
            cells[r, c] = np.abs(a[ys, xs] - b[ys, xs]).mean() > thresh
    half = grid // 2
    return {
        "changed_fraction": round(float(cells.mean()), 2),
        "top_half_changed": round(float(cells[:half].mean()), 2),
        "bottom_half_changed": round(float(cells[half:].mean()), 2),
    }


# ---------------------------------------------------------------- strips

def _font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_strips(small: np.ndarray, events: list[dict], fps: float, out_dir: Path) -> list[str]:
    """One row per event: frames around the boundary, with the frame where the new shot starts outlined."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("strip_*.jpg"):
        old.unlink()
    font, small_font = _font(15), _font(12)
    n = STRIP_BEFORE + STRIP_AFTER
    pad, label_h, row_label_h = 6, 16, 22
    row_h = row_label_h + SMALL_H + label_h + pad
    width = n * (SMALL_W + pad) + pad
    files = []
    for page_idx in range(0, len(events), STRIPS_PER_PAGE):
        page = events[page_idx:page_idx + STRIPS_PER_PAGE]
        img = Image.new("RGB", (width, row_h * len(page) + pad), (24, 24, 24))
        draw = ImageDraw.Draw(img)
        for r, ev in enumerate(page):
            y0 = pad + r * row_h
            draw.text((pad, y0 + 2), ev["label"], fill=(255, 220, 90), font=font)
            first_new = ev["new_shot_frame"]
            for c, f in enumerate(range(first_new - STRIP_BEFORE, first_new + STRIP_AFTER)):
                x = pad + c * (SMALL_W + pad)
                y = y0 + row_label_h
                if 0 <= f < len(small):
                    img.paste(Image.fromarray(small[f]), (x, y))
                if ev["lo"] <= f <= ev["hi"] + 1:
                    color = (90, 200, 255) if f != first_new else (255, 80, 80)
                    draw.rectangle([x - 2, y - 2, x + SMALL_W + 1, y + SMALL_H + 1], outline=color, width=2)
                draw.text((x + 2, y + SMALL_H + 1), f"{f} {f / fps:.2f}s", fill=(200, 200, 200), font=small_font)
        path = out_dir / f"strip_{page_idx // STRIPS_PER_PAGE + 1:02d}.jpg"
        img.save(path, quality=88)
        files.append(rel(path))
    return files


# ---------------------------------------------------------------- main analysis

def analyse(video: Path, threshold: float) -> dict:
    info = probe_video(video)
    fps = info["fps"]
    out_dir = analysis_dir_for(video)
    transcript = load_json(out_dir / "transcript.json")
    words = transcript["words"] if transcript else []

    single, many = run_transnet(video)
    small = decode_frames(video, SMALL_W, SMALL_H)
    n = min(len(single), len(small))
    single, many, small = single[:n], many[:n], small[:n]
    diff, luma = frame_signals(small)

    # --- boundaries -> transitions
    transitions = []
    for run_lo, run_hi in find_boundaries(single, threshold):
        peak = run_lo + int(np.argmax(single[run_lo:run_hi + 1]))
        lo, hi = transition_span(many, peak)
        lo, hi = min(lo, run_lo), max(hi, run_hi)
        kind, metrics = classify(diff, luma, lo, hi)
        # For a hard cut the new shot starts on the frame with the biggest change.
        if kind == "cut":
            region = range(max(1, lo), min(n, hi + 2))
            new_shot_frame = max(region, key=lambda f: diff[f])
        else:
            new_shot_frame = hi + 1
        t = new_shot_frame / fps
        transitions.append({
            "time": round(t, 3),
            "new_shot_frame": int(new_shot_frame),
            "type": kind,
            "frames": int(hi - lo + 1) if kind != "cut" else 0,
            "duration": round((hi - lo + 1) / fps, 3) if kind != "cut" else 0.0,
            "span": [int(lo), int(hi)],
            "confidence": round(float(single[peak]), 3),
            **metrics,
            "speech": word_context(t, words),
            "near_text_boundary": near_boundary(t, words),
        })

    # --- shots
    starts = [0] + [tr["new_shot_frame"] for tr in transitions]
    ends = [tr["span"][0] - 1 if tr["type"] != "cut" else tr["new_shot_frame"] - 1 for tr in transitions] + [n - 1]
    shots = []
    for i, (s, e) in enumerate(zip(starts, ends)):
        e = max(e, s)
        t0, t1 = s / fps, (e + 1) / fps
        spoken = [w["display"] for w in words if t0 <= (w["start"] + w["end"]) / 2 < t1]
        shots.append({
            "index": i, "start_frame": int(s), "end_frame": int(e),
            "start": round(t0, 3), "end": round(t1, 3), "duration": round(t1 - t0, 3),
            "words": len(spoken), "text": " ".join(spoken),
        })

    # --- jump-cut candidates (not seen by TransNetV2)
    jumps = jump_cut_candidates(diff, [tr["new_shot_frame"] for tr in transitions], fps)
    for j in jumps:
        j.update(changed_area(small, j["frame"]))
        j["speech"] = word_context(j["time"], words)
        j["near_text_boundary"] = near_boundary(j["time"], words)

    # --- strips: every transition + every jump-cut candidate
    events = []
    for k, tr in enumerate(transitions, 1):
        ctx = tr["speech"]
        said = f'"{ctx.get("prev_word") or ""} | {ctx.get("next_word") or ctx.get("word") or ""}"' if ctx else ""
        events.append({**tr, "lo": tr["span"][0], "hi": tr["span"][1],
                       "label": f'#{k} {tr["type"].upper()} @ {tr["time"]:.2f}s  p={tr["confidence"]:.2f}  '
                                f'{tr["frames"]}f  {said}'})
    for j in jumps:
        ctx = j["speech"]
        said = f'"{ctx.get("prev_word") or ""} | {ctx.get("next_word") or ctx.get("word") or ""}"' if ctx else ""
        events.append({"new_shot_frame": j["frame"], "lo": j["frame"], "hi": j["frame"] - 1, "time": j["time"],
                       "label": f'JUMP? @ {j["time"]:.2f}s  x{j["ratio"]}  area {j["changed_fraction"]:.0%}  {said}'})
    events.sort(key=lambda e: e["time"])
    strips = render_strips(small, events, fps, out_dir / "strips")

    np.savez_compressed(out_dir / "shots_raw.npz", cut_prob=single, transition_prob=many, diff=diff, luma=luma)

    # --- stats
    durations = np.array([s["duration"] for s in shots])
    cuts = [tr for tr in transitions]
    by_type: dict[str, int] = {}
    for tr in cuts:
        by_type[tr["type"]] = by_type.get(tr["type"], 0) + 1
    positions: dict[str, int] = {}
    for tr in cuts:
        pos = tr["speech"].get("position", "no_transcript")
        positions[pos] = positions.get(pos, 0) + 1
    on_boundary: dict[str, int] = {}
    for tr in cuts:
        key = tr["near_text_boundary"] or "none"
        on_boundary[key] = on_boundary.get(key, 0) + 1
    minutes = n / fps / 60

    return {
        "source": rel(video),
        "video": info,
        "frames": n,
        "model": "TransNetV2 (transnetv2-pytorch)",
        "threshold": threshold,
        "stats": {
            "shot_count": len(shots),
            "cuts_per_minute": round(len(cuts) / minutes, 1) if minutes else None,
            "shot_duration": {
                "mean": round(float(durations.mean()), 3),
                "median": round(float(np.median(durations)), 3),
                "p10": round(float(np.percentile(durations, 10)), 3),
                "p90": round(float(np.percentile(durations, 90)), 3),
                "min": round(float(durations.min()), 3),
                "max": round(float(durations.max()), 3),
            },
            "transition_types": by_type,
            "cut_position_vs_speech": positions,
            "cut_near_text_boundary": on_boundary,
            "jump_cut_candidates": len(jumps),
            "first_cut_at": cuts[0]["time"] if cuts else None,
        },
        "transitions": transitions,
        "jump_cut_candidates": jumps,
        "shots": shots,
        "strips": strips,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", help="Videos or folders (default: all of reference/)")
    parser.add_argument("--threshold", type=float, default=0.5, help="TransNetV2 cut probability threshold")
    parser.add_argument("--force", action="store_true", help="Re-run even if shots.json exists")
    args = parser.parse_args()

    videos = collect_videos(args.paths)
    if not videos:
        parser.error("no videos found")

    for video in videos:
        out_file = analysis_dir_for(video) / "shots.json"
        if out_file.exists() and not args.force:
            print(f"[skip] {video.name} (already analysed)")
            continue
        print(f"[shots] {video.name}", flush=True)
        result = analyse(video, args.threshold)
        save_json(out_file, result)
        s = result["stats"]
        d = s["shot_duration"]
        print(f"  -> {rel(out_file)}  {s['shot_count']} shots, {s['cuts_per_minute']} cuts/min, "
              f"median shot {d['median']}s, types {s['transition_types']}, "
              f"{s['jump_cut_candidates']} jump-cut candidates")
    return 0


if __name__ == "__main__":
    sys.exit(main())
