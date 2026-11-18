"""Colour and grade analysis: how the camera footage is graded, and which colours the graphics use.

Usage:
    uv run detect_colour.py                                   # every video under reference/
    uv run detect_colour.py reference/kallawaymarketing/2026-11-08_DffZRmfMWrf.mp4 --force

Uses shots.json + faces.json (run detect_shots.py and detect_faces.py first) to know each shot's
layout. Each sampled frame is split into
    camera    real footage: the whole frame for a talking head, the inset window in split layouts,
              b-roll without a face. This is where the grade lives.
    graphics  designed areas (the flat canvas, cards, screenshots). This is where brand colours live.

All measurements are in CIE Lab (L 0-100 = lightness, a = green(-)/magenta(+), b = blue(-)/yellow(+),
chroma = colour intensity, hue in degrees: ~40-70 skin/orange, ~200-250 teal/blue).

Writes to analysis/<creator>/<name>/:
    colour.json       grade (exposure, contrast, black/white point, saturation, white-balance cast,
                      shadow/highlight tint, skin tone, vignette), palettes, per-shot values, look tags
    colour_plot.png   frame barcode, grade over time, tone histogram, palette swatches
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import analysis_dir_for, collect_videos, load_json, probe_video, rel, save_json
from detect_faces import iter_frames

SAMPLE_W = 270              # analysis width (px); height keeps the aspect ratio
SAMPLES_PER_SECOND = 3
CANVAS_DIFF = 12.0          # Lab distance from the canvas colour that counts as "not canvas"
CAMERA_ROW_SHARE = 0.5      # a row/column belongs to the inset window if this share differs from canvas
GRAPHIC_CANVAS_SHARE = 0.4  # faceless frame with this much flat canvas = graphic, otherwise b-roll
PALETTE_K = 6
SPLIT_SUPPORT = 4.0         # border strength (x typical edge) that marks a window around a "talking head"
CAST_CHROMA = 12           # pixels below this chroma are "near neutral" (used to read white balance)


# ---------------------------------------------------------------- colour helpers

def to_lab(rgb_u8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb_u8.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def lab_to_hex(lab) -> str:
    rgb = cv2.cvtColor(np.array([[lab]], np.float32), cv2.COLOR_LAB2RGB)[0, 0]
    return "#{:02X}{:02X}{:02X}".format(*[int(round(float(np.clip(c, 0, 1)) * 255)) for c in rgb])


def hue(a: float, b: float) -> float:
    return float(np.degrees(np.arctan2(b, a)) % 360)


def colourfulness(rgb: np.ndarray) -> float:
    """Hasler & Suesstrunk colourfulness (0 = grey; ~30 moderate; 60+ very colourful)."""
    r, g, b = [rgb[:, i].astype(np.float32) for i in range(3)]
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


def grade_stats(lab: np.ndarray, rgb: np.ndarray) -> dict | None:
    """Grade numbers for a set of pixels (N x 3 Lab, N x 3 RGB)."""
    if len(lab) < 200:
        return None
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    C = np.hypot(a, b)
    p = np.percentile(L, [1, 5, 25, 50, 75, 95, 99])
    neutral = (C < CAST_CHROMA) & (L > 15) & (L < 90)
    lo, hi = L <= p[2], L >= p[4]

    def tint(mask):
        if mask.sum() < 50:
            return None
        ma, mb = float(a[mask].mean()), float(b[mask].mean())
        return {"a": round(ma, 2), "b": round(mb, 2), "chroma": round(float(np.hypot(ma, mb)), 2),
                "hue": round(hue(ma, mb), 1)}
    return {
        "exposure_L": round(float(L.mean()), 2),
        "median_L": round(float(p[3]), 2),
        "black_point_L": round(float(p[0]), 2),
        "white_point_L": round(float(p[6]), 2),
        "contrast_p95_p5": round(float(p[5] - p[1]), 2),
        "contrast_std": round(float(L.std()), 2),
        "clip_shadows_pct": round(float((L < 2).mean() * 100), 2),
        "clip_highlights_pct": round(float((L > 98).mean() * 100), 2),
        "chroma_mean": round(float(C.mean()), 2),
        "colourfulness": round(colourfulness(rgb), 2),
        "cast": tint(neutral),            # white balance: tint of near-neutral pixels
        "shadows_tint": tint(lo),         # split toning: average colour of the darkest quarter
        "highlights_tint": tint(hi),      # ... and of the brightest quarter
    }


def kmeans_palette(lab: np.ndarray, k: int = PALETTE_K, max_px: int = 40000) -> list[dict]:
    if len(lab) < k * 10:
        return []
    rng = np.random.default_rng(0)
    px = lab[rng.choice(len(lab), min(max_px, len(lab)), replace=False)].astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    _, labels, centres = cv2.kmeans(px, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    share = np.bincount(labels.ravel(), minlength=k) / len(labels)
    out = []
    for i in np.argsort(-share):
        c = centres[i]
        out.append({"hex": lab_to_hex(c), "share": round(float(share[i]), 3), "L": round(float(c[0]), 1),
                    "chroma": round(float(np.hypot(c[1], c[2])), 1), "hue": round(hue(c[1], c[2]), 1)})
    return out


# ---------------------------------------------------------------- frame regions

def canvas_colour(lab: np.ndarray) -> np.ndarray:
    """The flat background of split layouts: the median colour of the top rows."""
    return np.median(lab[: max(2, lab.shape[0] // 12)].reshape(-1, 3), axis=0)


def longest_run_containing(flags: np.ndarray, idx: int) -> tuple[int, int] | None:
    if not flags[idx]:
        near = np.nonzero(flags)[0]
        if not len(near):
            return None
        idx = int(near[np.argmin(np.abs(near - idx))])
    s = idx
    while s > 0 and flags[s - 1]:
        s -= 1
    e = idx
    while e + 1 < len(flags) and flags[e + 1]:
        e += 1
    return s, e + 1


def camera_mask(lab: np.ndarray, kind: str, face: np.ndarray | None,
                inset_box: tuple[int, int, int, int] | None = None) -> tuple[np.ndarray, str, tuple | None]:
    """Boolean mask of camera footage in the frame, the frame kind, and the inset box (r0, r1, c0, c1)
    found in this frame. `inset_box` is the video's typical inset window: split layouts are
    templates, so when this frame's own detection is much smaller (dark graphic above a dark
    inset, say) the typical window is used instead."""
    H, W = lab.shape[:2]
    full = np.ones((H, W), bool)
    if kind == "talking_head":
        return full, "talking_head", None
    canvas = canvas_colour(lab)
    differs = np.linalg.norm(lab - canvas, axis=2) > CANVAS_DIFF
    if kind == "face_inset" and face is not None and np.isfinite(face[0]):
        cx, cy = int(face[0] * W), int(face[1] * H)
        rows = longest_run_containing(differs.mean(axis=1) > CAMERA_ROW_SHARE, min(H - 1, cy))
        if rows:
            r0, r1 = rows
            cols = longest_run_containing(differs[r0:r1].mean(axis=0) > CAMERA_ROW_SHARE, min(W - 1, cx))
            found = (r0, r1, cols[0], cols[1]) if cols else None
        else:
            found = None
        box = inset_box or found
        m = np.zeros((H, W), bool)
        if box:
            m[box[0]:box[1], box[2]:box[3]] = True
        return m, "face_inset", found
    # no face: a designed graphic (lots of flat canvas) or b-roll footage
    if (~differs).mean() >= GRAPHIC_CANVAS_SHARE:
        return np.zeros((H, W), bool), "graphic", None
    return full, "broll", None


# ---------------------------------------------------------------- analysis

def skin_pixels(lab: np.ndarray, face: np.ndarray | None) -> np.ndarray:
    """Cheeks/nose area of the primary face, skipping dark beard/mouth/eye pixels."""
    if face is None or not np.isfinite(face[0]):
        return np.empty((0, 3), np.float32)
    H, W = lab.shape[:2]
    cx, cy, w, h = face[0] * W, face[1] * H, face[2] * W, face[3] * H
    x0, x1 = int(max(0, cx - 0.28 * w)), int(min(W, cx + 0.28 * w))
    y0, y1 = int(max(0, cy - 0.05 * h)), int(min(H, cy + 0.22 * h))
    px = lab[y0:y1, x0:x1].reshape(-1, 3)
    keep = (px[:, 0] > 25) & (px[:, 0] < 92) & (px[:, 1] > 2) & (px[:, 2] > 2)
    return px[keep]


def vignette(lab: np.ndarray) -> float:
    """Corner lightness relative to the centre (1.0 = flat, <0.85 = noticeable darkening)."""
    L = lab[..., 0]
    H, W = L.shape
    ch, cw = H // 6, W // 6
    corners = np.concatenate([L[:ch, :cw].ravel(), L[:ch, -cw:].ravel(), L[-ch:, :cw].ravel(), L[-ch:, -cw:].ravel()])
    centre = L[H // 3: 2 * H // 3, W // 3: 2 * W // 3]
    return float(corners.mean() / max(1e-3, centre.mean()))


def look_tags(g: dict, skin: dict | None) -> list[str]:
    """Plain-words description from the numbers (thresholds are rules of thumb)."""
    tags = []
    L = g["exposure_L"]
    tags.append("low-key / dark" if L < 30 else "bright / high-key" if L > 65 else "mid exposure")
    c = g["contrast_p95_p5"]
    tags.append("high contrast" if c > 75 else "low contrast / flat" if c < 45 else "moderate contrast")
    if g["black_point_L"] > 8:
        tags.append("lifted blacks (faded)")
    elif g["clip_shadows_pct"] > 5:
        tags.append("crushed blacks")
    if g["clip_highlights_pct"] > 2:
        tags.append("clipped highlights")
    ch = g["chroma_mean"]
    tags.append("muted / desaturated" if ch < 8 else "saturated" if ch > 22 else "natural saturation")
    cast = g.get("cast")
    if cast and cast["chroma"] > 2.5:
        h = cast["hue"]
        tags.append(("slightly " if cast["chroma"] < 5 else "") + ("warm cast" if 20 <= h <= 100 else "cool cast" if 180 <= h <= 290 else
                    "magenta cast" if h < 20 or h > 330 else "green cast" if 100 < h < 180 else "purple cast"))
    sh, hi = g.get("shadows_tint"), g.get("highlights_tint")
    if sh and hi and sh["chroma"] > 3 and hi["chroma"] > 3:
        if 170 <= sh["hue"] <= 280 and 20 <= hi["hue"] <= 100:
            tags.append("teal-orange split toning")
    if hi and hi["chroma"] > 6 and 20 <= hi["hue"] <= 100 and (not sh or sh["chroma"] < 4):
        tags.append("warm highlights, neutral shadows")
    if skin:
        tags.append(f"skin hue {skin['hue']:.0f}° ({'natural' if 35 <= skin['hue'] <= 70 else 'shifted'})")
    return tags


def median_stats(items: list[dict]) -> dict:
    """Element-wise median over a list of grade_stats dicts (nested tints included)."""
    items = [i for i in items if i]
    if not items:
        return {}
    out = {}
    for k, v in items[0].items():
        if isinstance(v, dict) or v is None:
            subs = [i[k] for i in items if i.get(k)]
            if subs:
                out[k] = {kk: round(float(np.median([s[kk] for s in subs])), 2) for kk in ("a", "b", "chroma")}
                out[k]["hue"] = round(hue(out[k]["a"], out[k]["b"]), 1)
            else:
                out[k] = None
        else:
            out[k] = round(float(np.median([i[k] for i in items if i.get(k) is not None])), 2)
    return out


def window_from_canvas(frames: list[tuple[int, np.ndarray, np.ndarray | None]]) -> tuple[int, int, int, int] | None:
    """Camera window on a plain canvas, wherever it sits: per frame, the connected region around
    the face that differs from the canvas colour; the window is the median of those boxes when
    they agree. Fails (None) on dark canvases, where dark footage blends into the canvas."""
    boxes = []
    for _, rgb, face in frames:
        if face is None or not np.isfinite(face[0]):
            continue
        lab = to_lab(rgb)
        H, W = lab.shape[:2]
        differs = (np.linalg.norm(lab - canvas_colour(lab), axis=2) > CANVAS_DIFF).astype(np.uint8)
        differs = cv2.morphologyEx(differs, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, lab_cc, st, _ = cv2.connectedComponentsWithStats(differs, connectivity=8)
        cx, cy = min(W - 1, int(face[0] * W)), min(H - 1, int(face[1] * H))
        k = lab_cc[cy, cx]
        if k == 0:
            continue
        x, y, w, h, area = st[k]
        if w * h > 0.8 * H * W or area < 0.5 * w * h:  # bled into the canvas, or not a window shape
            continue
        boxes.append((y, y + h, x, x + w))
    if len(boxes) < max(3, 0.3 * len(frames)):
        return None
    b = np.array(boxes, float)
    med = np.median(b, axis=0)
    H, W = frames[0][1].shape[:2]
    # the boxes must agree (a template), within ~4% of the frame
    if np.median(np.abs(b - med), axis=0).max() > 0.04 * max(H, W):
        return None
    return tuple(int(v) for v in med)


def find_inset_window(frames: list[tuple[int, np.ndarray, np.ndarray | None]]) -> tuple[int, int, int, int] | None:
    """Camera window of split layouts: on a plain canvas, the region around the face that differs
    from the canvas (any position/size); otherwise the template-border edge method below."""
    candidates = [b for b in (window_from_canvas(frames), find_window_edges(frames)) if b]
    if len(candidates) < 2:
        return candidates[0] if candidates else None
    gy, gx = persistent_edges(frames)
    return max(candidates, key=lambda b: border_support(b, gy, gx))


def persistent_edges(frames: list[tuple[int, np.ndarray, np.ndarray | None]]) -> tuple[np.ndarray, np.ndarray]:
    """Edge strength that persists through every split shot (median over shots of each shot's
    mean |vertical| and |horizontal| gradient): template borders survive, content doesn't."""
    H, W = frames[0][1].shape[:2]
    by_shot: dict[int, list] = {}
    for shot, rgb, _ in frames:
        by_shot.setdefault(shot, []).append(rgb)
    gys, gxs = [], []
    for rgbs in by_shot.values():
        gy, gx = np.zeros((H - 1, W)), np.zeros((H, W - 1))
        for rgb in rgbs:
            L = to_lab(rgb)[..., 0]
            gy += np.abs(np.diff(L, axis=0))
            gx += np.abs(np.diff(L, axis=1))
        gys.append(gy / len(rgbs))
        gxs.append(gx / len(rgbs))
    return np.median(gys, axis=0), np.median(gxs, axis=0)


def border_support(box: tuple[int, int, int, int], gy: np.ndarray, gx: np.ndarray) -> float:
    """How strongly a candidate window's sides sit on persistent edges (sides on the frame
    boundary don't count: the window may bleed off the frame)."""
    t, b, l, r = box
    H, W = gy.shape[0] + 1, gx.shape[1] + 1
    base = float(np.median(gy)) + float(np.median(gx)) + 1e-3
    sides = []
    if t > 2:
        sides.append(max(float(gy[y, l:r].mean()) for y in range(max(0, t - 3), min(H - 1, t + 2))))
    if b < H - 2:
        sides.append(max(float(gy[y, l:r].mean()) for y in range(max(0, b - 3), min(H - 1, b + 2))))
    if l > 2:
        sides.append(max(float(gx[t:b, x].mean()) for x in range(max(0, l - 3), min(W - 1, l + 2))))
    if r < W - 2:
        sides.append(max(float(gx[t:b, x].mean()) for x in range(max(0, r - 3), min(W - 1, r + 2))))
    return float(np.mean(sides)) / base if sides else 0.0


def find_window_edges(frames: list[tuple[int, np.ndarray, np.ndarray | None]]) -> tuple[int, int, int, int] | None:
    """Locate the camera window of split layouts as (top, bottom, left, right) rows/columns.

    `frames` are (shot index, rgb, face) for split-layout frames. The window is part of a
    template: its border is in every split shot, while the graphics around it change from shot
    to shot and the captions only cover the middle. So for every shot we average edge strength
    over its frames, score each row by a low percentile across the columns (a border runs the
    full width), and take the median over shots (a border is in all of them). Top = strongest
    such edge above the face; bottom/left/right likewise, or the frame edge if there's none."""
    faces = [f for _, _, f in frames if f is not None and np.isfinite(f[0])]
    if len(frames) < 3 or not faces:
        return None
    H, W = frames[0][1].shape[:2]
    by_shot: dict[int, list] = {}
    for shot, rgb, _ in frames:
        by_shot.setdefault(shot, []).append(rgb)
    row_prof, col_prof = [], []
    face = np.median(np.array(faces), axis=0)
    fx, fy, fw, fh = face[0] * W, face[1] * H, face[2] * W, face[3] * H
    for rgbs in by_shot.values():
        gy, gx = np.zeros((H - 1, W)), np.zeros((H, W - 1))
        for rgb in rgbs:
            L = to_lab(rgb)[..., 0]
            gy += np.abs(np.diff(L, axis=0))
            gx += np.abs(np.diff(L, axis=1))
        gy, gx = gy / len(rgbs), gx / len(rgbs)
        # skip the columns behind the head: heads often break out over the window's top border
        cols = [x for x in range(int(0.15 * W), int(0.85 * W)) if abs(x - fx) > 1.2 * fw]
        row_prof.append(np.percentile(gy[:, cols or slice(int(0.15 * W), int(0.85 * W))], 20, axis=1))
        col_prof.append(np.percentile(gx[int(fy + 0.6 * fh):H - 2], 20, axis=0))
    rows, cols = np.median(row_prof, axis=0), np.median(col_prof, axis=0)

    def strongest(profile: np.ndarray, lo: int, hi: int) -> tuple[int, bool]:
        """Index of the peak in profile[lo:hi], and whether it clearly stands out."""
        lo, hi = max(0, lo), min(len(profile), hi)
        if hi - lo < 3:
            return lo, False
        seg = profile[lo:hi]
        i = int(np.argmax(seg))
        return lo + i, bool(seg[i] > max(0.3, 4 * np.median(profile)))

    # top: the clear edge nearest above the face (titles higher up can be persistent edges too)
    # (down to the face centre: big heads often pop out over the window's top border)
    lo, hi = int(0.2 * H), int(fy)
    if hi - lo < 3:
        return None
    thr = max(0.3, 4 * np.median(rows), 0.4 * rows[lo:hi].max())
    above = [y for y in range(hi - 1, lo - 1, -1) if rows[y] > thr]
    if not above:
        return None
    top = above[0]
    bottom, ok = strongest(rows, int(fy + 0.8 * fh), H - 2)
    bottom = bottom + 1 if ok else H
    left, ok = strongest(cols, 0, int(fx - 0.8 * fw))
    left = left + 1 if ok else 0
    right, ok = strongest(cols, int(fx + 0.8 * fw), W - 2)
    right = right + 1 if ok else W
    return top + 1, bottom, left, right


def analyse(video: Path) -> dict:
    info = probe_video(video)
    fps, out_dir = info["fps"], analysis_dir_for(video)
    W = SAMPLE_W
    H = int(round(W * info["height"] / info["width"] / 2)) * 2
    shots = (load_json(out_dir / "shots.json") or {}).get("shots") or \
        [{"index": 0, "start_frame": 0, "end_frame": 10 ** 9, "start": 0, "end": info["duration"]}]
    face_shots = {s["index"]: s for s in (load_json(out_dir / "faces.json") or {}).get("shots", [])}
    raw = out_dir / "faces_raw.npz"
    primary = np.load(raw)["primary"] if raw.exists() else None
    step = max(1, round(fps / SAMPLES_PER_SECOND))
    # caption text sits on the footage: leave it out of the grade (it is white/black by design)
    caption_boxes: dict[int, list] = {}
    for c in (load_json(out_dir / "captions.json") or {}).get("captions", []):
        for f in range(c["start_frame"], c["end_frame"] + 1):
            caption_boxes.setdefault(f, []).extend(l["box"] for l in c["lines"])

    shot_of = lambda f: next((s for s in shots if s["start_frame"] <= f <= s["end_frame"]), shots[-1])
    sampled = []  # (frame, rgb, shot, layout, face)
    for f, bgr in enumerate(iter_frames(video, W, H)):
        if f % step:
            continue
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        shot = shot_of(f)
        layout = face_shots.get(shot["index"], {}).get("layout", "no_face")
        face = primary[f] if primary is not None and f < len(primary) else None
        sampled.append((f, rgb, shot, layout, face))
    inset_box = find_inset_window([(shot["index"], rgb, face) for _, rgb, shot, layout, face in sampled
                                   if layout == "face_inset"])
    # A big face in a big window gets labelled "talking_head" by the face step. Look for a window
    # border around those shots too, and treat a shot as split layout when its frames sit inside a
    # strong persistent border (full-frame footage has no such straight edge: support ~1).
    split_box, split_shots = None, set()
    th = [(shot["index"], rgb, face) for _, rgb, shot, layout, face in sampled if layout == "talking_head"]
    if th:
        cand = find_inset_window(th)
        if cand and (cand[1] - cand[0]) * (cand[3] - cand[2]) < 0.75 * H * W:
            for idx in {s for s, _, _ in th}:
                fr = [x for x in th if x[0] == idx]
                gy, gx = persistent_edges(fr)
                if border_support(cand, gy, gx) >= SPLIT_SUPPORT:
                    split_shots.add(idx)
            split_box = cand if split_shots else None

    frames, cam_lab, cam_rgb, gfx_lab, skin_lab, barcode, canvases = [], [], [], [], [], [], []
    for f, rgb, shot, layout, face in sampled:
        lab = to_lab(rgb)
        box = inset_box
        if shot["index"] in split_shots:
            layout, box = "face_inset", split_box
        cam, kind, _ = camera_mask(lab, layout, face, box)
        for x0, y0, x1, y1 in caption_boxes.get(f, []):
            pad = 0.25 * (y1 - y0)
            cam[max(0, int((y0 - pad) * H)):int((y1 + pad) * H) + 1,
                max(0, int((x0 - pad) * W)):int((x1 + pad) * W) + 1] = False
        rec = {"frame": f, "time": round(f / fps, 3), "shot": shot["index"], "kind": kind,
               "camera_share": round(float(cam.mean()), 3)}
        if cam.any():
            cl, cr = lab[cam], rgb[cam]
            rec["camera"] = grade_stats(cl, cr)
            if rec["camera"] and kind in ("talking_head", "broll"):
                rec["camera"]["edge_to_centre"] = round(vignette(lab), 3)
            cam_lab.append(cl[::7]); cam_rgb.append(cr[::7])
        if (~cam).any():
            gfx_lab.append(lab[~cam][::7])
        if kind in ("face_inset", "graphic"):
            canvases.append(canvas_colour(lab))
        sk = skin_pixels(lab, face)
        if len(sk) > 30:
            skin_lab.append(sk)
            rec["skin"] = [round(float(x), 2) for x in np.median(sk, axis=0)]
        barcode.append(cv2.resize(rgb, (1, 160), interpolation=cv2.INTER_AREA)[:, 0])
        frames.append(rec)

    # the grade is read from footage of the creator; faceless b-roll / screen content is reported
    # separately (it is often screens or stock with its own look)
    face_frames = [r["camera"] for r in frames if r.get("camera") and r["kind"] in ("talking_head", "face_inset")]
    broll_frames = [r["camera"] for r in frames if r.get("camera") and r["kind"] == "broll"]
    cam_frames = face_frames or broll_frames
    camera = median_stats(cam_frames)
    broll = median_stats(broll_frames) if face_frames and broll_frames else None
    if camera:
        # corner/centre lightness; can't tell a grading vignette from a dark room, so not a tag
        vig = [r["camera"]["edge_to_centre"] for r in frames if r.get("camera", {}) and r["camera"].get("edge_to_centre")]
        camera["edge_to_centre"] = round(float(np.median(vig)), 3) if vig else None
    skin = None
    if skin_lab:
        s = np.median(np.concatenate(skin_lab), axis=0)
        skin = {"hex": lab_to_hex(s), "L": round(float(s[0]), 1), "chroma": round(float(np.hypot(s[1], s[2])), 1),
                "hue": round(hue(s[1], s[2]), 1)}

    per_shot = []
    for s in shots:
        rs = [r for r in frames if r["shot"] == s["index"]]
        if not rs:
            continue
        kinds = [r["kind"] for r in rs]
        per_shot.append({"index": s["index"], "start": s["start"], "end": s["end"],
                         "kind": max(set(kinds), key=kinds.count),
                         "camera": median_stats([r.get("camera") for r in rs])})
    # how consistent is the grade from shot to shot (camera footage only)?
    cam_shots = [p["camera"] for p in per_shot if p["camera"]]
    consistency = None
    if len(cam_shots) > 1:
        consistency = {
            "exposure_L_std": round(float(np.std([c["exposure_L"] for c in cam_shots])), 2),
            "chroma_std": round(float(np.std([c["chroma_mean"] for c in cam_shots])), 2),
            "cast_b_std": round(float(np.std([c["cast"]["b"] for c in cam_shots if c.get("cast")] or [0])), 2),
        }
    gfx = np.concatenate(gfx_lab) if gfx_lab else np.empty((0, 3), np.float32)
    gfx_palette = kmeans_palette(gfx, 8)
    kinds = [r["kind"] for r in frames]
    result = {
        "source": rel(video),
        "sampled_frames": len(frames),
        "frame_kinds": {k: round(kinds.count(k) / len(kinds), 3) for k in sorted(set(kinds))},
        "inset_window": {"top": round(inset_box[0] / H, 3), "bottom": round(inset_box[1] / H, 3),
                         "left": round(inset_box[2] / W, 3), "right": round(inset_box[3] / W, 3)}
        if inset_box else None,
        "split_window": {"top": round(split_box[0] / H, 3), "bottom": round(split_box[1] / H, 3),
                         "left": round(split_box[2] / W, 3), "right": round(split_box[3] / W, 3),
                         "shots": sorted(split_shots),
                         "note": "large face window: the face step called these shots talking_head"}
        if split_box else None,
        "camera_grade": camera,
        "broll_grade": broll,
        "skin": skin,
        "look": look_tags(camera, skin) if camera else [],
        "camera_palette": kmeans_palette(np.concatenate(cam_lab)) if cam_lab else [],
        "graphics": {
            "canvas": lab_to_hex(np.median(canvases, axis=0)) if canvases else None,
            "palette": gfx_palette,
            "accents": [p for p in gfx_palette if p["chroma"] > 30 and p["share"] > 0.005],
        },
        "consistency": consistency,
        "shots": per_shot,
        "frames": frames,
    }
    plot(result, np.stack(barcode, axis=1), np.concatenate(cam_lab) if cam_lab else None, out_dir / "colour_plot.png")
    return result


# ---------------------------------------------------------------- plot

def plot(res: dict, barcode: np.ndarray, cam_lab: np.ndarray | None, path: Path) -> None:
    fr = res["frames"]
    t = [r["time"] for r in fr]
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(4, 3, height_ratios=[1.2, 1, 1, 0.5])
    ax = fig.add_subplot(gs[0, :])
    ax.imshow(barcode, aspect="auto", extent=[t[0], t[-1], 1, 0])
    ax.set_title("frame barcode (each column = one sampled frame squeezed to 1 px wide)")
    ax.set_yticks([])
    ax = fig.add_subplot(gs[1, :])
    cam_t = [r["time"] for r in fr if r.get("camera")]
    get = lambda k: [r["camera"][k] for r in fr if r.get("camera")]
    ax.plot(cam_t, get("exposure_L"), ".", ms=3, label="exposure L")
    ax.plot(cam_t, get("contrast_p95_p5"), ".", ms=3, label="contrast (p95-p5)")
    ax.plot(cam_t, get("chroma_mean"), ".", ms=3, label="chroma")
    ax.plot(cam_t, [r["camera"]["cast"]["b"] if r["camera"].get("cast") else np.nan for r in fr if r.get("camera")],
            ".", ms=3, label="cast b (+warm)")
    ax.set_xlim(t[0], t[-1]); ax.legend(loc="upper right", fontsize=8); ax.set_title("camera footage grade over time")
    ax.grid(alpha=0.3)
    ax = fig.add_subplot(gs[2, 0])
    if cam_lab is not None:
        ax.hist(cam_lab[:, 0], bins=50, range=(0, 100), color="0.4")
    ax.set_title("camera tone histogram (L)"); ax.set_xlabel("lightness")
    ax = fig.add_subplot(gs[2, 1])
    if cam_lab is not None:
        sub = cam_lab[:: max(1, len(cam_lab) // 4000)]
        cols = cv2.cvtColor(sub.reshape(-1, 1, 3).astype(np.float32), cv2.COLOR_LAB2RGB).reshape(-1, 3)
        ax.scatter(sub[:, 1], sub[:, 2], c=np.clip(cols, 0, 1), s=2)
    ax.axhline(0, color="0.7", lw=0.5); ax.axvline(0, color="0.7", lw=0.5)
    ax.set_xlim(-40, 40); ax.set_ylim(-40, 40); ax.set_title("camera vectorscope (a/b)")
    ax.set_xlabel("a  (green - / magenta +)"); ax.set_ylabel("b  (blue - / yellow +)")
    ax = fig.add_subplot(gs[2, 2]); ax.axis("off")
    ax.text(0, 1, "look:\n  " + "\n  ".join(res["look"]), va="top", fontsize=9, family="monospace")
    for i, (name, pal) in enumerate((("camera palette", res["camera_palette"]),
                                     ("graphics palette", res["graphics"]["palette"]))):
        ax = fig.add_subplot(gs[3, i]); ax.set_title(name, fontsize=9); ax.axis("off")
        x = 0
        for p in pal:
            ax.add_patch(plt.Rectangle((x, 0), p["share"], 1, color=p["hex"]))
            if p["share"] > 0.06:
                ax.text(x + 0.005, 0.4, p["hex"], fontsize=7, color="white" if p["L"] < 55 else "black")
            x += p["share"]
        ax.set_xlim(0, max(1, x)); ax.set_ylim(0, 1)
    if res["skin"]:
        ax = fig.add_subplot(gs[3, 2]); ax.axis("off"); ax.set_title("skin (median)", fontsize=9)
        ax.add_patch(plt.Rectangle((0, 0), 0.3, 1, color=res["skin"]["hex"]))
        ax.text(0.35, 0.4, f'{res["skin"]["hex"]}  hue {res["skin"]["hue"]}°', fontsize=8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", help="Videos or folders (default: all of reference/)")
    parser.add_argument("--force", action="store_true", help="Re-run even if colour.json exists")
    args = parser.parse_args()
    videos = collect_videos(args.paths)
    if not videos:
        parser.error("no videos found")
    for video in videos:
        out_file = analysis_dir_for(video) / "colour.json"
        if out_file.exists() and not args.force:
            print(f"[skip] {video.name} (already analysed)")
            continue
        print(f"[colour] {video.name}", flush=True)
        res = analyse(video)
        save_json(out_file, res)
        g = res["camera_grade"]
        print(f"  -> {rel(out_file)}  kinds {res['frame_kinds']}\n     L {g.get('exposure_L')}  contrast "
              f"{g.get('contrast_p95_p5')}  chroma {g.get('chroma_mean')}  look: {', '.join(res['look'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
