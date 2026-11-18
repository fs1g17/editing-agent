"""Identify a creator's caption font: measure it, shortlist free fonts, verify by rendering.

Usage:
    uv run match_font.py kallawaymarketing              # needs captions.json for the creator's videos
    uv run match_font.py kallawaymarketing --vlm        # also ask a VLM for candidate fonts
    uv run match_font.py kallawaymarketing --also "Inter,Archivo"   # always score these families

Needs models/font_index.json (uv run font_index.py).

Steps
  1. measure  From the caption crops alone (no font needed): x-height, cap height, stroke
              thickness, letter gaps and word spaces, as ratios of the x-height.
  2. shortlist  Keep Google Fonts instances whose stroke weight and x-height/cap-height ratio are
              close to the measured ones (plus any VLM / --also suggestions).
  3. verify   Render each caption's text in each candidate, fit size and letter spacing
              (word width is linear in letter spacing, so it's solved, not searched), and score
              the pixel overlap with the real caption (soft IoU, 1.0 = identical).

Writes analysis/<creator>/font/:
    font.json     measurements, ranked candidates (score, weight, size, letter spacing)
    compare.png   real captions vs the best matches, with overlap colouring
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from common import ANALYSIS_DIR, ROOT, load_json, probe_video, rel, save_json
from detect_captions import split_text
from fontlib import EM, FONTS_DIR, INDEX_PATH, FontInstance, font_axes, ink_box, stroke_width

X_LETTERS = set("acemnorsuvwxz")         # letters that sit exactly between baseline and x-height
STROKE_TOL = 0.22                         # shortlist: stroke/x-height within this fraction
RATIO_TOL = 0.06                          # shortlist: x-height/cap-height within this
QUICK_CROPS, FULL_CROPS = 8, 30           # captions used in the quick pass / the full pass
QUICK_KEEP = 80                           # instances surviving the quick pass
TRACK_LIMITS = (-0.15, 0.3)               # letter spacing in em; outside this the font is wrong anyway


# ---------------------------------------------------------------- crops -> text coverage

def letters(s: str) -> str:
    return "".join(c for c in s if c.isalpha())


def text_coverage(crop: np.ndarray, box_px: tuple[float, float, float, float]) -> dict | None:
    """Soft text mask (0..1) of the caption line, tight around the ink. Handles captions on a
    backing box/pill (the text is split from the box, see detect_captions.split_text)."""
    H, W = crop.shape[:2]
    x0, y0, x1, y1 = box_px
    lh = y1 - y0
    X0, X1 = max(0, int(x0 - 0.15 * lh)), min(W, int(x1 + 0.15 * lh))
    Y0, Y1 = max(0, int(y0 - 0.12 * lh)), min(H, int(y1 + 0.12 * lh))
    region_rgb = np.ascontiguousarray(crop[Y0:Y1, X0:X1])
    if region_rgb.size < 300:
        return None
    text, box = split_text(region_rgb)
    region = cv2.cvtColor(region_rgb, cv2.COLOR_RGB2GRAY)
    if text.mean() < 0.02 or text.mean() > 0.7:
        return None
    near = cv2.dilate(text.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    back = (box & ~near) if box is not None else ~near
    if back.sum() < 50:
        return None
    fg, bg = float(np.median(region[text])), float(np.median(region[back]))
    if abs(fg - bg) < 60 or np.std(region[back]) > 25:  # low contrast or busy background
        return None
    cov = np.clip((region.astype(np.float32) - bg) / (fg - bg), 0, 1)
    if box is not None:
        inside = cv2.erode(box.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        cov = np.where(inside, cov, 0.0)
    hard = (cov >= 0.5).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(hard, connectivity=8)
    rh, rw = hard.shape
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        touches = x == 0 or y == 0 or x + w >= rw or y + h >= rh
        if area >= 3 and not touches:
            keep[i] = True
    if not keep.any():
        return None
    kept = keep[lab]
    near = cv2.dilate(kept.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    cov = np.where(near, cov, 0.0)
    b = ink_box(cov)
    if b is None:
        return None
    return {"cov": cov[b[1]:b[3], b[0]:b[2]], "hard": kept[b[1]:b[3], b[0]:b[2]],
            "components": int(keep.sum()), "on_box": box is not None}


def load_crops(creator: str) -> list[dict]:
    out = []
    for cj in sorted((ANALYSIS_DIR / creator).glob("*/captions.json")):
        data = load_json(cj)
        info = probe_video(ROOT / data["source"])
        W, H = info["width"], info["height"]
        for cap in data["captions"]:
            if cap.get("line_count") != 1 or cap.get("emphasis") or not cap.get("crop"):
                continue
            text = cap["lines"][0]["text"]
            # use only captions whose OCR text agrees with what was said (no misreads)
            if letters(text).lower() != letters(cap.get("spoken", "")).lower() or not letters(text):
                continue
            path = ROOT / cap["crop"].replace("\\", "/")
            if not path.exists():
                continue
            crop = np.asarray(Image.open(path).convert("RGB"))
            x0, y0, x1, y1 = cap["lines"][0]["box"]
            pad_y = (y1 - y0) * 0.25  # same geometry detect_captions.py used to cut the crop
            CX0, CY0 = max(0, int((x0 - 0.03) * W)), max(0, int((y0 - pad_y) * H))
            box = (x0 * W - CX0, y0 * H - CY0, x1 * W - CX0, y1 * H - CY0)
            tc = text_coverage(crop, box)
            # the ink must span most of the OCR box (or of the pill around it), otherwise part of
            # the caption was lost
            if tc is None or tc["cov"].shape[1] < (0.55 if tc["on_box"] else 0.8) * (box[2] - box[0]):
                continue
            out.append({"video": cj.parent.name, "text": text, "start": cap["start"], "crop": crop, **tc})
    return out


# ---------------------------------------------------------------- 1. measure

def ink_runs(hard: np.ndarray) -> list[tuple[int, int]]:
    cols = hard.any(axis=0)
    runs, start = [], None
    for x, on in enumerate(cols):
        if on and start is None:
            start = x
        elif not on and start is not None:
            runs.append((start, x)); start = None
    if start is not None:
        runs.append((start, len(cols)))
    return runs


X_FLAT = set("xzvwnmru")       # x-height letters with (nearly) flat tops/bottoms
CAP_FLAT = set("EFHIKLMNTXZ")   # capitals with flat tops/bottoms
ASC_FLAT = set("bdhkl")         # lowercase ascenders


def slant(hard: np.ndarray) -> float | None:
    """Slant of the letters in degrees (positive = tops lean right, as in italics): the shear that
    makes vertical strokes most vertical, i.e. the column profile sharpest."""
    h, w = hard.shape
    if h < 12 or w < 2 * h:
        return None
    m = hard.astype(np.float32)
    best, best_a = -1.0, 0.0
    for a in range(-20, 21):
        t = np.tan(np.radians(a))
        M = np.float32([[1, t, h - t * h], [0, 1, 0]])  # x' = x - t * (h - y) + h: undo a lean of angle a
        sheared = cv2.warpAffine(m, M, (w + 2 * h, h), flags=cv2.INTER_NEAREST)
        cols = sheared.sum(axis=0)
        sharp = float((cols ** 2).sum())
        if sharp > best:
            best, best_a = sharp, float(a)
    return best_a


def measure(crops: list[dict]) -> dict:
    """Font-agnostic measurements. Works per caption: when every character is its own column of
    ink (letters don't touch), each letter's height and the gaps between letters are readable,
    and everything is expressed relative to that caption's own x-height."""
    per_video: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for c in crops:
        t, h = c["text"], c["hard"]
        chars = t.replace(" ", "")
        runs = ink_runs(h)
        if len(runs) != len(chars):
            continue
        heights = []
        for a, b in runs:
            ys = np.nonzero(h[:, a:b].any(axis=1))[0]
            heights.append(ys.max() - ys.min() + 1)
        xs = [hh for ch, hh in zip(chars, heights) if ch in X_FLAT]
        if not xs:
            continue
        xh = float(np.median(xs))
        pv = per_video[c["video"]]
        pv["x_height_px"].append(xh)
        pv["stroke_to_x"].append(stroke_width(h) / xh)
        pv["cap_to_x"] += [hh / xh for ch, hh in zip(chars, heights) if ch in CAP_FLAT]
        pv["ascender_to_x"] += [hh / xh for ch, hh in zip(chars, heights) if ch in ASC_FLAT]
        breaks, pos = set(), 0
        for w in t.split(" ")[:-1]:
            pos += len(w)
            breaks.add(pos - 1)
        for i in range(len(runs) - 1):
            gap = (runs[i + 1][0] - runs[i][1]) / xh
            if i in breaks:
                pv["word_space_to_x"].append(gap)
            elif chars[i].isalpha() and chars[i + 1].isalpha():
                pv["letter_gap_to_x"].append(gap)

    def med(v):
        return round(float(np.median(v)), 4) if v else None
    keys = ("x_height_px", "stroke_to_x", "cap_to_x", "ascender_to_x", "letter_gap_to_x", "word_space_to_x")
    pool = {k: [x for d in per_video.values() for x in d[k]] for k in keys}
    m = {k: med(v) for k, v in pool.items()}
    m["x_to_cap"] = round(1 / m["cap_to_x"], 3) if m["cap_to_x"] else None
    slants = [a for c in crops if len(letters(c["text"])) >= 5 and (a := slant(c["hard"])) is not None]
    m["slant_deg"] = round(float(np.median(slants)), 1) if slants else None
    m["italic"] = bool(m["slant_deg"] is not None and m["slant_deg"] >= 5)
    if m["letter_gap_to_x"] and m["word_space_to_x"]:
        m["word_space_to_letter_gap"] = round(m["word_space_to_x"] / m["letter_gap_to_x"], 2)
    m["x_height_px_range"] = [round(float(np.percentile(pool["x_height_px"], q)), 1) for q in (10, 90)] \
        if pool["x_height_px"] else None
    m["samples"] = {k: len(v) for k, v in pool.items()}
    m["per_video"] = {v: {k: med(d[k]) for k in keys} for v, d in per_video.items()}
    return m


# ---------------------------------------------------------------- 3. verify by rendering

def soft_iou(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.minimum(a, b).sum() / max(1e-6, np.maximum(a, b).sum()))


def fit(inst: FontInstance, crop: dict) -> dict | None:
    """Render the caption text in `inst`, fit size + letter spacing to the crop, score overlap."""
    text, T = crop["text"], crop["cov"]
    glyphs = inst.shape(text)
    if glyphs is None:
        return None
    base = inst.render(text, 0.0, glyphs)
    b = ink_box(base)
    if b is None:
        return None
    w0, h0 = b[2] - b[0], b[3] - b[1]
    Th, Tw = T.shape
    n_gaps = len(glyphs) - 1
    track = 0.0
    if n_gaps:
        track = (Tw * h0 / Th - w0) / (n_gaps * EM)
        track = float(np.clip(track, *TRACK_LIMITS))
    r = inst.render(text, track, glyphs) if track else base
    b = ink_box(r)
    R = cv2.resize(r[b[1]:b[3], b[0]:b[2]], (Tw, Th), interpolation=cv2.INTER_AREA)
    # Compare letter shapes at the 50% edge (robust to the video's blur), allowing a 1 px nudge.
    pad = 2
    Tp = np.pad(T >= 0.5, pad)
    best = 0.0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            Rp = np.zeros_like(Tp)
            Rp[pad + dy:pad + dy + Th, pad + dx:pad + dx + Tw] = R >= 0.5
            best = max(best, (Tp & Rp).sum() / max(1, (Tp | Rp).sum()))
    width_err = abs((b[2] - b[0]) * Th / h0 - Tw) / Tw  # >0 only when tracking hit its limit
    return {"iou": float(best), "track_em": track, "size_px": EM * Th / h0, "width_err": width_err, "render": R}


def score(inst: FontInstance, crops: list[dict]) -> dict | None:
    fits = [fit(inst, c) for c in crops]
    if any(f is None for f in fits):
        return None
    ious = [f["iou"] for f in fits]
    keep = sorted(ious)[len(ious) // 5:]  # drop the worst 20%: stray non-caption crops shouldn't decide
    return {"score": round(float(np.mean(keep)), 4), "mean": round(float(np.mean(ious)), 4),
            "median": round(float(np.median(ious)), 4),
            "worst": round(float(np.min(ious)), 4),
            "track_em": round(float(np.median([f["track_em"] for f in fits])), 4),
            "track_spread": round(float(np.std([f["track_em"] for f in fits])), 4),
            "size_px": round(float(np.median([f["size_px"] for f in fits])), 1),
            "_fits": fits}


def typical_word_count(crops: list[dict]) -> int:
    return Counter(len(c["text"].split()) for c in crops).most_common(1)[0][0]


def pick_crops(crops: list[dict], n: int) -> list[dict]:
    """Captions in the creator's usual format (e.g. one word at a time; longer lines are often
    title graphics in another font), with varied letters, spread across the videos."""
    usual = typical_word_count(crops)
    seen, by_video = set(), defaultdict(list)
    for c in sorted(crops, key=lambda c: -len(set(letters(c["text"]).lower()))):
        key = c["text"].lower()
        if len(letters(c["text"])) < 4 or key in seen or len(c["text"].split()) != usual:
            continue
        seen.add(key)
        by_video[c["video"]].append(c)
    out, i = [], 0
    while len(out) < n and any(i < len(v) for v in by_video.values()):
        for v in by_video.values():
            if i < len(v) and len(out) < n:
                out.append(v[i])
        i += 1
    return out


def instance_name(e: dict) -> str:
    wd = e["axes"].get("wdth")
    return (f'{e["family"]} {e["weight"]}' + (" Italic" if e.get("italic") else "")
            + (f" wdth {wd:g}" if wd is not None else "")
            + (f" + {e['oblique_deg']:g}deg slant" if e.get("oblique_deg") else ""))


def make_instance(e: dict) -> FontInstance:
    return FontInstance(FONTS_DIR / e["file"], e["axes"], e.get("oblique_deg", 0.0))


def rank(entries: list[dict], crops: list[dict], label: str) -> list[dict]:
    t0, out = time.time(), []
    for i, e in enumerate(entries):
        try:
            s = score(make_instance(e), crops)
        except Exception as ex:
            print(f"  skip {instance_name(e)}: {ex}", file=sys.stderr)
            continue
        if s:
            out.append(e | s)
        if i % 200 == 199:
            print(f"  {label}: {i + 1}/{len(entries)}  {time.time() - t0:.0f}s", flush=True)
    out.sort(key=lambda r: -r["score"])
    print(f"  {label}: {len(entries)} instances x {len(crops)} captions in {time.time() - t0:.0f}s")
    return out


def refine(best: dict, crops: list[dict]) -> dict:
    """Variable fonts: search weight between the 100-steps and, if the font has a width axis,
    its width too (letter spacing alone can't make a wider font's letters narrower)."""
    axes = {a[0]: a for a in font_axes(FONTS_DIR / best["file"])}
    if "wght" not in axes:
        return best
    _, wlo, _, whi = axes["wght"]
    weights = [w for w in range(int(best["weight"]) - 75, int(best["weight"]) + 76, 25) if wlo <= w <= whi]
    widths = [None]
    if "wdth" in axes:
        _, dlo, dflt, dhi = axes["wdth"]
        widths = [d for d in (dflt + k * 5 for k in range(-6, 4)) if dlo <= d <= dhi]
    top = best
    for w in weights:
        for d in widths:
            var = best["axes"] | {"wght": float(w)} | ({"wdth": float(d)} if d is not None else {})
            if var == best["axes"]:
                continue
            s = score(make_instance(best | {"axes": var}), crops)
            if s and s["score"] > top["score"]:
                top = best | {"weight": w, "axes": var} | s
    return top


def degrade(cov: np.ndarray, height: int) -> np.ndarray:
    """Make a clean render look like a caption cut from a compressed 1080p video."""
    h, w = cov.shape
    small = cv2.resize(cov, (max(1, round(w * height / h)), height), interpolation=cv2.INTER_AREA)
    pad = 6
    img = np.pad(small, pad)
    img = cv2.GaussianBlur(img, (0, 0), 0.7)
    ok, enc = cv2.imencode(".jpg", (img * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 60])
    img = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
    b = ink_box(img)
    return img[b[1]:b[3], b[0]:b[2]]


def calibrate(results: list[dict], crops: list[dict]) -> dict:
    """What would a perfect match score? Fake the captions in the best match's own font (same
    sizes, spacing, blur and compression as the real ones), then score the best match and the
    runners-up against those. Same font -> the ceiling; other fonts -> how separable they are."""
    truth = results[0]
    synth = []
    for c, f in zip(crops, truth["_fits"]):
        inst = make_instance(truth)
        r = inst.render(c["text"], f["track_em"])
        b = ink_box(r)
        synth.append({"text": c["text"], "cov": degrade(r[b[1]:b[3], b[0]:b[2]], c["cov"].shape[0])})
    out = {"reference_font": instance_name(truth), "same_font": score(make_instance(truth), synth)["score"]}
    out["other_fonts"] = {instance_name(r): score(make_instance(r), synth)["score"] for r in results[1:6]}
    return out


# ---------------------------------------------------------------- report image

def compare_image(results: list[dict], crops: list[dict], path: Path, n_crops: int = 8, n_fonts: int = 3) -> None:
    rows = crops[:n_crops]
    th = 70
    cols = [("real", None)] + [(instance_name(r), r) for r in results[:n_fonts]]
    cell_w = 380
    img = Image.new("RGB", (cell_w * len(cols), 50 + len(rows) * (th + 30)), (24, 24, 24))
    d = ImageDraw.Draw(img)
    d.text((8, 32), "overlay: white = both agree, red = only in the real caption, cyan = only in the font render",
           fill=(150, 150, 150))
    img_off = 20
    for ci, (name, r) in enumerate(cols):
        title = name if r is None else f"{name}  ({r['score']:.3f})"
        d.text((ci * cell_w + 8, 8), title, fill=(255, 220, 90))
    for ri, c in enumerate(rows):
        y = 30 + img_off + ri * (th + 30)
        T = c["cov"]
        s = th / T.shape[0]
        size = (max(1, int(T.shape[1] * s)), th)
        real = (cv2.resize(T, size) * 255).astype(np.uint8)
        img.paste(Image.fromarray(real).convert("RGB").resize((min(size[0], cell_w - 16), th)), (8, y))
        d.text((8, y + th + 4), c["text"], fill=(180, 180, 180))
        for ci, (_, r) in enumerate(cols[1:], 1):
            f = r["_fits"][ri]
            Rr = cv2.resize(f["render"], size)
            Tr = cv2.resize(T, size)
            # white = both, red = only real, cyan = only render
            over = np.stack([Tr, Rr, Rr], -1)
            tile = Image.fromarray((np.clip(over, 0, 1) * 255).astype(np.uint8))
            img.paste(tile.resize((min(size[0], cell_w - 16), th)), (ci * cell_w + 8, y))
            d.text((ci * cell_w + 8, y + th + 4), f"iou {f['iou']:.3f}  track {f['track_em']:+.3f}em",
                   fill=(180, 180, 180))
    img.save(path)


# ---------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("creator")
    parser.add_argument("--also", default="", help="comma-separated families to always score")
    parser.add_argument("--vlm", action="store_true", help="ask a VLM for candidates (Vercel AI Gateway)")
    args = parser.parse_args()
    creator = Path(args.creator).name

    index = load_json(INDEX_PATH)
    if not index:
        parser.error("no font index - run font_index.py first")
    crops = load_crops(creator)
    if not crops:
        parser.error(f"no usable caption crops under analysis/{creator}/ - run detect_captions.py first")
    print(f"{len(crops)} clean caption crops")

    m = measure(crops)
    print(f"measured: x-height {m['x_height_px']}px  x/cap {m['x_to_cap']}  ascender/x {m['ascender_to_x']}  "
          f"stroke/x {m['stroke_to_x']}  letter gap/x {m['letter_gap_to_x']}  word space/x {m['word_space_to_x']}")

    # 2. shortlist
    insts = index["instances"]
    for e in insts:
        e["x_to_cap"] = e["xh"] / e["cap"]
    short = [e for e in insts
             if abs(e["stroke"] - m["stroke_to_x"]) <= STROKE_TOL * m["stroke_to_x"]
             and (m["x_to_cap"] is None or abs(e["x_to_cap"] - m["x_to_cap"]) <= RATIO_TOL)]
    if m["italic"]:
        # slanted captions: true italics, plus upright fonts slanted by the measured angle
        # (editing apps' "italic" button usually just shears the upright font)
        short = [e for e in short if e.get("italic")] +                 [e | {"oblique_deg": m["slant_deg"]} for e in short if not e.get("italic")]
        print(f"captions lean {m['slant_deg']} deg: testing true italics and slanted uprights")
    else:
        short = [e for e in short if not e.get("italic")]
    suggested: dict[str, str] = {f.strip().lower(): "--also" for f in args.also.split(",") if f.strip()}
    vlm = None
    if args.vlm:
        from vlm_fonts import ask_vlm
        vlm = ask_vlm(crops, m)
        for name in vlm.get("google_fonts", []):
            suggested.setdefault(name.lower(), "vlm")
    in_short = {(e["file"], json.dumps(e["axes"], sort_keys=True), e.get("oblique_deg")) for e in short}
    extra = [e | ({"oblique_deg": m["slant_deg"]} if m["italic"] and not e.get("italic") else {})
             for e in insts if e["family"].lower() in suggested and bool(e.get("italic")) <= m["italic"]]
    extra = [e for e in extra if (e["file"], json.dumps(e["axes"], sort_keys=True), e.get("oblique_deg")) not in in_short]
    print(f"shortlist: {len(short)} of {len(insts)} instances by weight/proportions "
          f"({len({e['family'] for e in short})} families) + {len(extra)} suggested")

    quick, full = pick_crops(crops, QUICK_CROPS), pick_crops(crops, FULL_CROPS)
    stage1 = rank(short + extra, quick, "quick pass")
    keep = stage1[:QUICK_KEEP] + [r for r in stage1[QUICK_KEEP:] if r["family"].lower() in suggested]
    stage2 = rank(keep, full, "full pass")
    best_by_family: dict[str, dict] = {}
    for r in stage2:
        best_by_family.setdefault(r["family"], r)
    families = sorted(best_by_family.values(), key=lambda r: -r["score"])
    print("  refining weight/width of the top 5 ...", flush=True)
    families[:5] = sorted((refine(r, full) for r in families[:5]), key=lambda r: -r["score"])
    cal = calibrate(families, full)

    out_dir = ANALYSIS_DIR / creator / "font"
    out_dir.mkdir(parents=True, exist_ok=True)
    compare_image(families, full, out_dir / "compare.png")
    clean = lambda r: {k: v for k, v in r.items() if not k.startswith("_") and k not in ("xh", "cap", "stroke", "x_to_cap")} | \
        {"suggested_by": suggested.get(r["family"].lower())}
    report = {
        "creator": creator,
        "measured": m,
        "captions_scored": [c["text"] for c in full],
        "best": clean(families[0]) | {
            "font_size_px_at_source": families[0]["size_px"],
            "letter_spacing_css": f'{families[0]["track_em"]:+.3f}em',
        },
        "families": [clean(r) for r in families[:20]],
        "suggestions": {name: next((clean(r) for r in families if r["family"].lower() == name), "not in index / failed")
                        for name in suggested},
        "calibration": cal,
        "vlm": vlm,
        "shortlist_size": len(short),
    }
    save_json(out_dir / "font.json", report)
    print("\nbest families (score = mean overlap, 1.0 identical):")
    for r in families[:10]:
        print(f"  {r['score']:.3f}  {instance_name(r):32s} size {r['size_px']:5.1f}px  "
              f"spacing {r['track_em']:+.3f}em  worst {r['worst']:.3f}")
    print(f"\ncalibration: the same font, faked through blur + compression, scores {cal['same_font']:.3f}; "
          f"against it the runners-up score " + ", ".join(f"{v:.3f}" for v in cal["other_fonts"].values()))
    print(f"-> {rel(out_dir / 'font.json')}\n-> {rel(out_dir / 'compare.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
