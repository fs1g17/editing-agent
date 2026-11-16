"""Build a caption track: what text is on screen, when, where, how big, what colour, how it animates,
and how it lines up with the spoken words. Also logs other on-screen text (graphics, UI).

Usage:
    uv run detect_captions.py reference/kallawaymarketing [--force]

Needs transcript.json (transcribe.py) - captions are told apart from other text by matching the speech.

How it works:
    pass 1  OCR the whole frame every SAMPLE_STEP frames; text matching nearby spoken words = captions.
            Their boxes give the caption band (the strip of the frame captions live in).
    pass 2  OCR just that band on every frame for frame-accurate timing and entry/exit animation.
    pass 3  full-resolution frames for colour (fill / outline / box / two-tone) and font crops.

Writes to analysis/<creator>/<name>/:
    captions.json                 caption events + style stats + other on-screen text
    captions/crops/NNN_text.png   full-res crop of every caption (input for font matching)
    captions/sheet_NN.jpg         contact sheets of the crops with their measurements, for checking
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import analysis_dir_for, collect_videos, load_json, probe_video, rel, save_json
from detect_faces import iter_frames

OCR_W = 720                    # OCR runs on frames resized to this width (full res misses text)
SAMPLE_STEP = 6                # pass 1: every Nth frame
MIN_SCORE = 0.6                # OCR confidence to keep a line
MATCH_RATIO = 0.75             # fuzzy similarity for an OCR token to count as a spoken word
MATCH_WINDOW = (-1.2, 0.6)     # spoken words starting this far before/after the frame are candidates (s)
MAX_CAPTION_WORDS = 8          # longer lines are graphics/UI text even if they echo the speech
ROW_BIN = 0.01                 # caption rows: histogram bin of line centre y (fraction of frame height)
ROW_MIN_SHARE = 0.08           # a row must hold matching text in this share of speech samples
BAND_MARGIN = 0.03             # grow the caption band by this fraction of frame height
FLICKER_GAP = 2                # frames: same caption interrupted this briefly is one caption
MIN_FRAMES = 2                 # captions shorter than this are OCR noise
EMPHASIS_RATIO = 1.4           # line this much taller than the typical caption line = emphasis text


# ---------------------------------------------------------------- OCR

def make_engine(side: int):
    from rapidocr import RapidOCR
    return RapidOCR(params={"Global.log_level": "error", "Global.use_cls": False,
                            "Det.limit_type": "max", "Det.limit_side_len": side})


def ocr_lines(engine, img: np.ndarray, y_offset: int, frame_w: int, frame_h: int) -> list[dict]:
    r = engine(img)
    if r.boxes is None:
        return []
    out = []
    for box, txt, score in zip(r.boxes, r.txts, r.scores):
        if score < MIN_SCORE or not txt.strip():
            continue
        b = np.array(box, np.float32)
        x0, x1 = b[:, 0].min() / frame_w, b[:, 0].max() / frame_w
        y0, y1 = (b[:, 1].min() + y_offset) / frame_h, (b[:, 1].max() + y_offset) / frame_h
        out.append({"text": txt.strip(), "score": round(float(score), 3),
                    "box": [round(float(x0), 4), round(float(y0), 4), round(float(x1), 4), round(float(y1), 4)]})
    return out


# ---------------------------------------------------------------- matching to speech

def norm(s: str) -> str:
    return re.sub(r"[^\w]", "", s.lower())


def match_span(text: str, t: float, words: list[dict]) -> tuple[list[dict], float]:
    """The run of consecutive spoken words that best matches the OCR text, ignoring spaces.
    Tightly tracked caption fonts make the OCR drop spaces ("Ican", "magicpipe"), so matching
    has to work on the letters, not on space-separated tokens."""
    target = norm(text)
    if not target:
        return [], 0.0
    best, best_r = [], 0.0
    for a, w in enumerate(words):
        if not (MATCH_WINDOW[0] <= w["start"] - t <= MATCH_WINDOW[1]):
            continue
        joined = ""
        for b in range(a, min(a + MAX_CAPTION_WORDS, len(words))):
            joined += norm(words[b]["word"])
            r = difflib.SequenceMatcher(None, target, joined, autojunk=False).ratio()
            if r > best_r:
                best, best_r = words[a:b + 1], r
            if len(joined) > 1.5 * len(target) + 3:
                break
    return best, best_r


def resegment(text: str, span: list[dict]) -> str:
    """Put back spaces the OCR dropped, at the word boundaries of the matched speech.
    Keeps the OCR's own spelling, case and punctuation (that's what is on screen)."""
    parts = [norm(w["word"]) for w in span]
    bounds, acc = set(), 0
    for p in parts[:-1]:
        acc += len(p)
        bounds.add(acc)
    pos = [k for k, c in enumerate(text) if re.match(r"\w", c)]
    ocr_n = "".join(text[k].lower() for k in pos)
    insert_before = set()
    for a, b, size in difflib.SequenceMatcher(None, ocr_n, "".join(parts), autojunk=False).get_matching_blocks():
        for k in range(size):
            if b + k in bounds:
                insert_before.add(pos[a + k])
    out = []
    for k, c in enumerate(text):
        if k in insert_before and out and not out[-1].isspace():
            out.append(" ")
        out.append(c)
    return "".join(out)


def caption_span(text: str, t: float, words: list[dict]) -> list[dict] | None:
    """The spoken words this OCR line shows, or None if it isn't a caption."""
    span, ratio = match_span(text, t, words)
    if not span or ratio < MATCH_RATIO or len(span) > MAX_CAPTION_WORDS:
        return None
    return span


def is_caption(text: str, t: float, words: list[dict]) -> bool:
    return caption_span(text, t, words) is not None


def caption_rows(samples: list[dict], words: list[dict]) -> list[tuple[float, float]]:
    """Captions sit at a consistent height; graphics text that happens to echo the speech doesn't.
    Histogram the centre-y of speech-matching lines over sample frames and keep the busy rows."""
    speaking = [s for s in samples if any(w["start"] - 0.3 <= s["time"] <= w["end"] + 0.3 for w in words)]
    bins: dict[int, set] = {}
    for s in samples:
        for l in s["lines"]:
            if l["caption"]:
                cy = (l["box"][1] + l["box"][3]) / 2
                bins.setdefault(int(cy / ROW_BIN), set()).add(s["frame"])
    need = max(4, ROW_MIN_SHARE * len(speaking))
    keep = sorted(b for b, frames in bins.items() if len(frames) >= need)
    rows, cur = [], []
    for b in keep:
        if cur and b - cur[-1] > 2:
            rows.append(cur)
            cur = []
        cur.append(b)
    if cur:
        rows.append(cur)
    out = []
    for r in rows:
        lo, hi = r[0] * ROW_BIN, (r[-1] + 1) * ROW_BIN
        members = [l for s in samples for l in s["lines"]
                   if l["caption"] and lo <= (l["box"][1] + l["box"][3]) / 2 < hi]
        top = float(np.percentile([l["box"][1] for l in members], 5))
        bottom = float(np.percentile([l["box"][3] for l in members], 95))
        out.append((round(top, 4), round(bottom, 4)))
    return out


def in_rows(box: list[float], rows: list[tuple[float, float]], slack: float = 0.02) -> bool:
    cy = (box[1] + box[3]) / 2
    return any(top - slack <= cy <= bottom + slack for top, bottom in rows)


# ---------------------------------------------------------------- colour

def _median_rgb(px: np.ndarray) -> list[int]:
    return [int(v) for v in np.median(px, axis=0)] if len(px) else [0, 0, 0]


def hexcol(rgb) -> str:
    return "#{:02X}{:02X}{:02X}".format(*[int(c) for c in rgb])


def caption_colours(frame_rgb: np.ndarray, box: list[float]) -> dict:
    """Split the caption box into text vs background, then look for an outline ring and a backing box."""
    H, W = frame_rgb.shape[:2]
    x0, y0, x1, y1 = box
    pad = max(4, int((y1 - y0) * H * 0.15))
    X0, Y0 = max(0, int(x0 * W) - pad), max(0, int(y0 * H) - pad)
    X1, Y1 = min(W, int(x1 * W) + pad), min(H, int(y1 * H) + pad)
    crop = frame_rgb[Y0:Y1, X0:X1]
    if crop.size == 0:
        return {}
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    _, hi = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hi = hi > 0
    border = np.zeros_like(hi)
    border[:3], border[-3:], border[:, :3], border[:, -3:] = True, True, True, True
    # The text class is the one that rarely touches the crop border.
    text = hi if hi[border].mean() < 0.5 else ~hi
    t8 = text.astype(np.uint8)
    k3 = np.ones((3, 3), np.uint8)
    core = cv2.erode(t8, k3).astype(bool)
    if core.sum() < 20:
        core = text
    fill = _median_rgb(crop[core])
    stroke_px = max(2, int((Y1 - Y0) * 0.06))
    ring = cv2.dilate(t8, np.ones((stroke_px * 2 + 1,) * 2, np.uint8)).astype(bool) & ~text
    outer = cv2.dilate(t8, np.ones((stroke_px * 6 + 1,) * 2, np.uint8)).astype(bool) & \
        ~cv2.dilate(t8, np.ones((stroke_px * 3 + 1,) * 2, np.uint8)).astype(bool)
    ring_rgb, outer_rgb = _median_rgb(crop[ring]), _median_rgb(crop[outer])
    lum = lambda c: 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
    ring_std = float(np.std(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)[ring])) if ring.any() else 99
    stroke = abs(lum(ring_rgb) - lum(outer_rgb)) > 30 and ring_std < 45 and \
        abs(lum(ring_rgb) - lum(fill)) > 60
    # Backing box: background inside the padded crop is flat and differs from the frame around it.
    bg = ~cv2.dilate(t8, np.ones((stroke_px * 2 + 1,) * 2, np.uint8)).astype(bool)
    bg_px = crop[bg]
    bg_std = float(np.std(bg_px.mean(axis=1))) if len(bg_px) else 99
    big_pad = pad * 4
    BX0, BY0 = max(0, X0 - big_pad), max(0, Y0 - big_pad)
    BX1, BY1 = min(W, X1 + big_pad), min(H, Y1 + big_pad)
    surround = frame_rgb[BY0:BY1, BX0:BX1].copy().reshape(-1, 3)
    mask = np.ones((BY1 - BY0, BX1 - BX0), bool)
    mask[Y0 - BY0:Y1 - BY0, X0 - BX0:X1 - BX0] = False
    surround_rgb = _median_rgb(frame_rgb[BY0:BY1, BX0:BX1][mask])
    bg_rgb = _median_rgb(bg_px)
    backing = bg_std < 14 and abs(lum(bg_rgb) - lum(surround_rgb)) > 35
    # Highlight colour (e.g. active word in yellow/green): strongly saturated text pixels.
    # Skin from a hand behind the caption is only mildly saturated, so it doesn't count.
    highlight = None
    px = crop[core]
    if len(px) > 60:
        hsv = cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
        vivid = (hsv[:, 1] > 140) & (hsv[:, 2] > 115)
        if vivid.mean() > 0.15:
            highlight = hexcol(_median_rgb(px[vivid]))
    return {
        "fill": hexcol(fill),
        "outline": hexcol(ring_rgb) if stroke else None,
        "backing_box": hexcol(bg_rgb) if backing else None,
        "highlight": highlight,
        "text_is_lighter": bool(lum(fill) > lum(outer_rgb)),
    }


# ---------------------------------------------------------------- events

def case_style(text: str) -> str:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "none"
    if all(c.isupper() for c in letters):
        return "upper"
    if all(c.islower() for c in letters):
        return "lower"
    words = [w for w in text.split() if any(c.isalpha() for c in w)]
    if all(next(c for c in w if c.isalpha()).isupper() for w in words):
        return "title"
    return "mixed"


def build_events(per_frame: list[list[dict]], fps: float) -> list[dict]:
    """Runs of frames showing the same caption text -> events (flicker-tolerant)."""
    keys = [" / ".join(norm(l["text"]) for l in sorted(lines, key=lambda l: l["box"][1])) for lines in per_frame]
    runs = []  # [key, start, end]
    for f, k in enumerate(keys):
        if not k:
            continue
        if runs and runs[-1][0] == k and f - runs[-1][2] <= FLICKER_GAP + 1:
            runs[-1][2] = f
        else:
            runs.append([k, f, f])
    # A one- or two-frame misread inside a caption (K, K', K) is OCR noise: fold it back in.
    merged = []
    for run in runs:
        if (len(merged) >= 2 and merged[-2][0] == run[0] and merged[-1][2] - merged[-1][1] + 1 <= FLICKER_GAP
                and run[1] - merged[-2][2] <= FLICKER_GAP + 2):
            merged.pop()
            merged[-1][2] = run[2]
        else:
            merged.append(run)
    runs = merged
    # a run that is a prefix-state of the next (word appearing inside a growing caption) stays separate on purpose
    return [{"key": k, "start_frame": s, "end_frame": e} for k, s, e in runs if e - s + 1 >= MIN_FRAMES]


def summarise_event(ev: dict, per_frame: list[list[dict]], words: list[dict], fps: float) -> dict:
    s, e = ev["start_frame"], ev["end_frame"]
    frames = [per_frame[f] for f in range(s, e + 1) if per_frame[f]]
    # most common spelling per line position
    n_lines = Counter(len(fr) for fr in frames).most_common(1)[0][0]
    stable = [sorted(fr, key=lambda l: l["box"][1]) for fr in frames if len(fr) == n_lines]
    lines = []
    for i in range(n_lines):
        texts = Counter(fr[i]["text"] for fr in stable).most_common(1)[0][0]
        boxes = np.array([fr[i]["box"] for fr in stable])
        med = np.median(boxes, axis=0)
        lines.append({"text": texts, "box": [round(float(v), 4) for v in med],
                      "height": round(float(med[3] - med[1]), 4),
                      "center_x": round(float((med[0] + med[2]) / 2), 3),
                      "center_y": round(float((med[1] + med[3]) / 2), 3)})
    text = " ".join(l["text"] for l in lines)
    t0 = s / fps
    matched = match_span(text, t0, words)[0]
    first = min(matched, key=lambda w: w["start"]) if matched else None
    last = max(matched, key=lambda w: w["end"]) if matched else None

    # entry / exit animation: main line height and position over the first/last frames vs its steady state
    main = max(range(n_lines), key=lambda i: lines[i]["height"])
    def track(fr_range):
        out = []
        for f in fr_range:
            fr = sorted(per_frame[f], key=lambda l: l["box"][1]) if 0 <= f < len(per_frame) else []
            if len(fr) == n_lines:
                b = fr[main]["box"]
                out.append((round((b[3] - b[1]) / lines[main]["height"], 2),
                            round((b[1] + b[3]) / 2 - lines[main]["center_y"], 3)))
            else:
                out.append(None)
        return out
    entry = track(range(s, min(s + 6, e + 1)))
    exit_ = track(range(max(s, e - 5), e + 1))

    def anim(seq):
        vals = [v for v in seq if v]
        if len(vals) < 2:
            return "none"
        # OCR boxes jitter by ~10% frame to frame, so an animation has to hold for 2+ frames
        # and move one way (growing into / shrinking out of the steady size).
        scales = [v[0] for v in vals]
        ys = [v[1] for v in vals]
        if scales[0] <= 0.9 and scales[1] <= 0.97 and scales[1] >= scales[0]:
            return "scale"
        if scales[-1] <= 0.9 and scales[-2] <= 0.97 and scales[-2] >= scales[-1]:
            return "scale"
        if abs(ys[0]) > 0.01 and abs(ys[1]) > 0.005 and abs(ys[1]) < abs(ys[0]):
            return "slide"
        if abs(ys[-1]) > 0.01 and abs(ys[-2]) > 0.005 and abs(ys[-2]) < abs(ys[-1]):
            return "slide"
        return "none"

    return {
        "text": text,
        "start_frame": s, "end_frame": e,
        "start": round(t0, 3), "end": round((e + 1) / fps, 3), "duration": round((e - s + 1) / fps, 3),
        "word_count": len(text.split()),
        "line_count": n_lines,
        "lines": lines,
        "case": case_style(text),
        "has_punctuation": bool(re.search(r"[.,!?;:]", text)),
        "spoken": " ".join(m["display"] for m in matched),
        "first_word_start": first["start"] if first else None,
        "lead_s": round(first["start"] - t0, 3) if first else None,       # >0: caption shows before the word
        "linger_s": round((e + 1) / fps - last["end"], 3) if last else None,  # >0: stays after the word ends
        "entry_animation": anim(entry), "entry_track": entry,
        "exit_animation": anim(exit_), "exit_track": exit_,
    }


# ---------------------------------------------------------------- sheets

def _font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def save_crops_and_sheets(events: list[dict], crops: dict[int, np.ndarray], out_dir: Path) -> None:
    crop_dir = out_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    for old in list(crop_dir.glob("*.png")) + list(out_dir.glob("sheet_*.jpg")):
        old.unlink()
    tiles = []
    for i, ev in enumerate(events):
        img = crops.get(i)
        if img is None:
            continue
        safe = re.sub(r"[^\w]+", "_", ev["text"])[:30].strip("_") or "text"
        path = crop_dir / f"{i:03d}_{safe}.png"
        Image.fromarray(img).save(path)
        ev["crop"] = rel(path)
        tiles.append((ev, img))
    font = _font(14)
    tile_w, tile_h, label_h, per_page = 360, 120, 34, 40
    cols = 4
    for p in range(0, len(tiles), per_page):
        page = tiles[p:p + per_page]
        rows = (len(page) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * tile_w, rows * (tile_h + label_h)), (40, 40, 40))
        d = ImageDraw.Draw(sheet)
        for k, (ev, img) in enumerate(page):
            x, y = (k % cols) * tile_w, (k // cols) * (tile_h + label_h)
            im = Image.fromarray(img)
            im.thumbnail((tile_w - 8, tile_h - 4))
            sheet.paste(im, (x + 4, y + label_h))
            c = ev.get("colour", {})
            d.text((x + 4, y + 2), f'{ev["start"]:.2f}s {ev["duration"]:.2f}s lead {ev["lead_s"]}', fill=(255, 220, 90), font=font)
            d.text((x + 4, y + 17), f'h {ev["lines"][0]["height"]:.3f}  {c.get("fill")}  ol {c.get("outline")}  '
                                    f'{ev["entry_animation"]}', fill=(200, 200, 200), font=font)
        sheet.save(out_dir / f"sheet_{p // per_page + 1:02d}.jpg", quality=88)


# ---------------------------------------------------------------- main

def analyse(video: Path) -> dict:
    info = probe_video(video)
    fps = info["fps"]
    out_dir = analysis_dir_for(video)
    transcript = load_json(out_dir / "transcript.json")
    if not transcript:
        raise RuntimeError("no transcript.json - run transcribe.py first")
    words = transcript["words"]
    W = OCR_W
    H = int(round(W * info["height"] / info["width"] / 2)) * 2

    # pass 1: sparse full-frame OCR
    eng_full = make_engine(720)
    samples = []
    for f, frame in enumerate(iter_frames(video, W, H)):
        if f % SAMPLE_STEP:
            continue
        t = f / fps
        lines = ocr_lines(eng_full, frame, 0, W, H)
        for l in lines:
            l["caption"] = is_caption(l["text"], t, words)
        samples.append({"frame": f, "time": round(t, 3), "lines": lines})
    n_frames = f + 1

    rows = caption_rows(samples, words)
    if not rows:
        return {"source": rel(video), "captions": [], "stats": {"caption_count": 0},
                "note": "no on-screen text consistently matched the speech"}
    # lines outside the caption rows are graphics text, even if they echo the speech
    for s in samples:
        for l in s["lines"]:
            l["caption"] = l["caption"] and in_rows(l["box"], rows)
    band = (max(0.0, min(r[0] for r in rows) - BAND_MARGIN), min(1.0, max(r[1] for r in rows) + BAND_MARGIN))
    by0, by1 = int(band[0] * H), int(band[1] * H)

    # pass 2: every frame, band only
    eng_band = make_engine(480)
    per_frame: list[list[dict]] = []
    for f, frame in enumerate(iter_frames(video, W, H)):
        lines = ocr_lines(eng_band, frame[by0:by1], by0, W, H)
        t = f / fps
        kept = []
        for l in lines:
            if not in_rows(l["box"], rows):
                continue
            span = caption_span(l["text"], t, words)
            if span:
                l["ocr_raw"] = l["text"]
                l["text"] = resegment(l["text"], span)
                kept.append(l)
        per_frame.append(kept)

    events = [summarise_event(ev, per_frame, words, fps) for ev in build_events(per_frame, fps)]
    # Box height depends on letter shapes ("now" vs "why" vs "Claude"), so compare each line
    # with the typical height of lines that have the same ascender/descender profile.
    def profile(text: str) -> tuple[bool, bool]:
        return (bool(re.search(r"[A-Z0-9bdfhklt'\"]", text)), bool(re.search(r"[gjpqy,;]", text)))
    by_profile: dict[tuple, list[float]] = {}
    for ev in events:
        for l in ev["lines"]:
            by_profile.setdefault(profile(l["text"]), []).append(l["height"])
    typical_h = float(np.median([l["height"] for ev in events for l in ev["lines"]])) if events else 0
    typical_by_profile = {k: float(np.median(v)) for k, v in by_profile.items() if len(v) >= 5}
    for ev in events:
        ev["emphasis"] = any(
            l["height"] >= EMPHASIS_RATIO * typical_by_profile.get(profile(l["text"]), typical_h) for l in ev["lines"])

    # pass 3: full-res frames for colour + crops (middle frame of each caption)
    want = {}
    for i, ev in enumerate(events):
        want.setdefault((ev["start_frame"] + ev["end_frame"]) // 2, []).append(i)
    crops = {}
    for f, frame in enumerate(iter_frames(video, info["width"], info["height"])):
        if f not in want:
            continue
        rgb = frame[:, :, ::-1]
        for i in want[f]:
            ev = events[i]
            x0 = min(l["box"][0] for l in ev["lines"]); x1 = max(l["box"][2] for l in ev["lines"])
            y0 = min(l["box"][1] for l in ev["lines"]); y1 = max(l["box"][3] for l in ev["lines"])
            main = max(ev["lines"], key=lambda l: l["height"])
            ev["colour"] = caption_colours(rgb, main["box"])
            pad_y = (y1 - y0) * 0.25
            Y0, Y1 = max(0, int((y0 - pad_y) * info["height"])), min(info["height"], int((y1 + pad_y) * info["height"]))
            X0, X1 = max(0, int((x0 - 0.03) * info["width"])), min(info["width"], int((x1 + 0.03) * info["width"]))
            crops[i] = np.ascontiguousarray(rgb[Y0:Y1, X0:X1])
    save_crops_and_sheets(events, crops, out_dir / "captions")

    # other on-screen text (pass 1 lines that are not captions), deduplicated
    other: dict[str, dict] = {}
    for s in samples:
        for l in s["lines"]:
            if l["caption"]:
                continue
            k = norm(l["text"])
            if len(k) < 2:
                continue
            o = other.setdefault(k, {"text": l["text"], "first_seen": s["time"], "last_seen": s["time"],
                                     "box": l["box"], "height": round(l["box"][3] - l["box"][1], 4), "samples": 0})
            o["last_seen"] = s["time"]
            o["samples"] += 1

    # stats
    speech = [(w["start"], w["end"]) for w in words]
    covered = 0.0
    for a, b in speech:
        for ev in events:
            covered += max(0.0, min(b, ev["end"]) - max(a, ev["start"]))
    speech_total = sum(b - a for a, b in speech) or 1
    leads = [ev["lead_s"] for ev in events if ev["lead_s"] is not None]
    lingers = [ev["linger_s"] for ev in events if ev["linger_s"] is not None]
    normal = [ev for ev in events if not ev["emphasis"]]
    fills = Counter(ev.get("colour", {}).get("fill") for ev in normal)

    def share(pred, items):
        return round(sum(1 for x in items if pred(x)) / len(items), 2) if items else None

    stats = {
        "caption_count": len(events),
        "speech_coverage": round(min(1.0, covered / speech_total), 2),
        "caption_band_y": [round(band[0], 3), round(band[1], 3)],
        "caption_rows_y": [list(r) for r in rows],
        "words_per_caption": dict(sorted(Counter(ev["word_count"] for ev in events).items())),
        "lines_per_caption": dict(sorted(Counter(ev["line_count"] for ev in events).items())),
        "duration_median_s": round(float(np.median([ev["duration"] for ev in events])), 3) if events else None,
        "lead_median_s": round(float(np.median(leads)), 3) if leads else None,
        "linger_median_s": round(float(np.median(lingers)), 3) if lingers else None,
        "line_height": {  # box height as a fraction of frame height (font-size proxy)
            "median": round(typical_h, 4),
            "px_at_1920": round(typical_h * 1920),
        },
        "center_y_median": round(float(np.median([l["center_y"] for ev in normal for l in ev["lines"]])), 3) if normal else None,
        "center_x_median": round(float(np.median([l["center_x"] for ev in normal for l in ev["lines"]])), 3) if normal else None,
        "case": dict(Counter(ev["case"] for ev in events)),
        "punctuation_share": share(lambda ev: ev["has_punctuation"], events),
        "fill_colours": {k: v for k, v in fills.most_common(5) if k},
        # light text on darker surroundings vs dark text on lighter surroundings
        "light_text_share": share(lambda ev: ev.get("colour", {}).get("text_is_lighter"), events),
        "outline_share": share(lambda ev: ev.get("colour", {}).get("outline"), normal),
        "backing_box_share": share(lambda ev: ev.get("colour", {}).get("backing_box"), normal),
        "highlight_share": share(lambda ev: ev.get("colour", {}).get("highlight"), normal),
        "emphasis_count": sum(ev["emphasis"] for ev in events),
        "entry_animation": dict(Counter(ev["entry_animation"] for ev in events)),
        "exit_animation": dict(Counter(ev["exit_animation"] for ev in events)),
    }
    return {
        "source": rel(video),
        "ocr": "RapidOCR PP-OCRv6 small",
        "frames": n_frames,
        "stats": stats,
        "captions": events,
        "other_text": sorted(other.values(), key=lambda o: o["first_seen"]),
        "pass1_samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", help="Videos or folders (default: all of reference/)")
    parser.add_argument("--force", action="store_true", help="Re-run even if captions.json exists")
    args = parser.parse_args()
    videos = collect_videos(args.paths)
    if not videos:
        parser.error("no videos found")
    for video in videos:
        out_file = analysis_dir_for(video) / "captions.json"
        if out_file.exists() and not args.force:
            print(f"[skip] {video.name} (already analysed)")
            continue
        print(f"[captions] {video.name}", flush=True)
        try:
            result = analyse(video)
        except RuntimeError as exc:
            print(f"  !! {exc}", file=sys.stderr)
            continue
        save_json(out_file, result)
        s = result["stats"]
        print(f"  -> {rel(out_file)}  {s['caption_count']} captions, coverage {s.get('speech_coverage')}, "
              f"words/caption {s.get('words_per_caption')}, lead {s.get('lead_median_s')}s, "
              f"height {s.get('line_height')}, fills {s.get('fill_colours')}, entry {s.get('entry_animation')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
