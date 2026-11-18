"""VLM pass: describe each shot's graphics, motion graphics/animations, icon styling and anything else
visually notable, anchored to what is said.

Usage:
    uv run vlm_shots.py reference/kallawaymarketing/2026-11-08_DffZRmfMWrf.mp4
    uv run vlm_shots.py reference/kallawaymarketing --force          # all videos, redo cached shots
    uv run vlm_shots.py reference/kallawaymarketing --shots 0,3      # just these shots

Needs timeline.json (build_timeline.py). Per shot the model gets:
    sheet     12 frames with timestamps; sampled densely at the start of the shot (entry animations)
              and wherever the graphics change most (frames spread by cumulative change), so
              animations are visible as a sequence
    keyframe  one sharp frame late in the shot (everything has animated in) to judge styling
    context   what is said with word times, layout, measured camera moves/zooms, on-screen text (OCR)

Writes analysis/<creator>/<name>/vlm/:
    shot_NN.json, sheet_NN.jpg, key_NN.jpg   per shot (cached; --force to redo)
    vlm.json                                 all shots + total cost
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

from common import ROOT, analysis_dir_for, collect_videos, load_json, probe_video, rel, save_json
from detect_faces import iter_frames

MODEL = "anthropic/claude-opus-5.5"
URL = "https://ai-gateway.vercel.sh/v1/chat/completions"
SHEET_FRAMES, SHEET_COLS = 12, 4
TILE_W = 300
KEY_W = 720
ENERGY_W = 108              # low-res width for measuring how much the picture changes
DENSE_START = (2, 4, 7, 11)  # extra frames (offsets from shot start) to catch entry animations
PARALLEL = 4
SEGMENT_MAX_S = 6.0         # longer shots are split into parts so 12 frames still catch the animations

PROMPT = """You are analysing the visual editing style of a short-form video creator so an automated editor can
reproduce it. Below is ONE shot from one of their videos.

Image 1: {n} frames from the shot in time order, each labelled with its time (seconds into the video
and +offset from the shot start). Frames are spaced unevenly: dense at the start and wherever the
picture changes, so compare consecutive frames to work out what moves/animates and how.
Image 2: one sharp frame from late in the shot, for judging styling details.

Measured context (from code, trust it):
{context}

Notes:
- Captions ({captions}) are already measured.
  Mention them only if something about them is unusual in this shot.
- On-screen text from OCR may contain misreads; treat odd spellings as OCR noise.
- If frames are too sparse to be sure about an animation, say so in "confidence" rather than guessing.
- Times: use seconds from the shot start.

Reply with JSON only:
{{
  "summary": "one sentence: what the viewer sees in this shot",
  "visual_type": "talking_head | split_graphic_top_face_bottom | full_screen_graphic | broll | screen_recording | other",
  "background": {{"description": "...", "colour": "#hex or words", "texture": "e.g. plain, dot grid, line grid, gradient, glow, noise, photo"}},
  "elements": [
    {{"kind": "logo | icon | screenshot | ui_card | title_text | label_text | number | arrow | line | shape | chart | photo | video | emoji | sticker | other",
      "what": "what it depicts (e.g. 'Claude logo', 'YouTube channel card')",
      "style": "how it is rendered: outline/stroke vs filled, stroke weight, colours, glow/shadow/gradient, corner radius, frame/border, size relative to screen",
      "position": "where on screen"}}
  ],
  "animations": [
    {{"element": "which element (or 'whole graphic', 'camera', 'transition')",
      "kind": "entry | exit | emphasis | continuous | transition",
      "technique": "e.g. fade, slide, scale pop, overshoot bounce, draw-on stroke, mask wipe, typewriter, stagger, zoom, pan, parallax, rotate, glow pulse, shine sweep, blur-in, counter, morph",
      "direction": "from where / to where, if any",
      "start_s": 0.0, "duration_s": 0.0,
      "easing": "linear | ease-out | ease-in-out | overshoot/spring | unclear",
      "trigger": "the spoken word/phrase it lines up with, if any",
      "confidence": "high | medium | low"}}
  ],
  "text_styles": [
    {{"text": "...", "role": "title | subtitle | label | highlight | number | other",
      "look": "font category/weight/case, colour, treatments (highlight bar, underline, box, outline, glow)"}}
  ],
  "icon_style": "how icons/logos are stylised in this shot (outline vs filled, stroke colour/weight, containers, glow), or null",
  "speech_link": "how what is shown relates to what is said at that moment",
  "standouts": ["anything else visually relevant or distinctive in this shot"]
}}"""


# ---------------------------------------------------------------- frame selection

def change_energy(video: Path, info: dict, timeline: dict) -> np.ndarray:
    """Per-frame amount of picture change, ignoring the caption band (it changes every word) and,
    in split layouts, the camera window (the person moving isn't a motion graphic)."""
    W = ENERGY_W
    H = int(round(W * info["height"] / info["width"] / 2)) * 2
    meta = timeline["meta"]
    band = (meta.get("captions") or {}).get("center_y_median")
    win = meta.get("inset_window")
    shot_layout = []
    for s in timeline["shots"]:
        shot_layout.append((round(s["start"] * info["fps"]), s["layout"]))
    energy, prev = [], None
    for f, bgr in enumerate(iter_frames(video, W, H)):
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        mask = np.ones_like(g, bool)
        if band:
            mask[int((band - 0.05) * H):int((band + 0.05) * H)] = False
        layout = next((l for start, l in reversed(shot_layout) if f >= start), None)
        if layout == "face_inset" and win:
            mask[int(win["top"] * H):int(win["bottom"] * H), int(win["left"] * W):int(win["right"] * W)] = False
        energy.append(0.0 if prev is None else float(np.abs(g - prev)[mask].mean()))
        prev = g
    return np.array(energy)


def pick_frames(start: int, end: int, energy: np.ndarray) -> list[int]:
    """SHEET_FRAMES frame numbers for a shot: first/last, a few right after the cut, the rest spread
    evenly over the shot's cumulative change (so busy moments get more frames)."""
    n = end - start + 1
    picks = {start, end}
    picks |= {start + o for o in DENSE_START if o < n}
    e = energy[start:end + 1].copy()
    e[0] = 0  # the cut itself
    remaining = SHEET_FRAMES - len(picks)
    if remaining > 0 and e.sum() > 0:
        cum = np.cumsum(e) / e.sum()
        for q in np.linspace(0, 1, remaining + 2)[1:-1]:
            picks.add(start + int(np.searchsorted(cum, q)))
    # top up evenly in time if change-based picks collided
    for f in np.linspace(start, end, SHEET_FRAMES + 2).astype(int)[1:-1]:
        if len(picks) >= SHEET_FRAMES:
            break
        picks.add(int(f))
    picks = sorted(picks)
    if len(picks) > SHEET_FRAMES:  # keep first and last, thin the middle evenly
        middle = picks[1:-1]
        idx = np.linspace(0, len(middle) - 1, SHEET_FRAMES - 2).round().astype(int)
        picks = [picks[0]] + [middle[i] for i in sorted(set(idx))] + [picks[-1]]
    return picks


def _font(size: int):
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_sheet(frames: dict[int, np.ndarray], picks: list[int], fps: float, shot_start: float) -> Image.Image:
    font = _font(18)
    tiles = []
    for f in picks:
        im = Image.fromarray(frames[f][:, :, ::-1])
        im = im.resize((TILE_W, round(TILE_W * im.height / im.width)), Image.LANCZOS)
        d = ImageDraw.Draw(im)
        t = f / fps
        label = f"{t:.2f}s (+{t - shot_start:.2f})"
        d.rectangle((0, 0, TILE_W, 24), fill=(0, 0, 0))
        d.text((6, 2), label, fill=(255, 220, 60), font=font)
        tiles.append(im)
    th = tiles[0].height
    rows = (len(tiles) + SHEET_COLS - 1) // SHEET_COLS
    sheet = Image.new("RGB", (SHEET_COLS * TILE_W + (SHEET_COLS - 1) * 6, rows * th + (rows - 1) * 6), (90, 90, 90))
    for i, im in enumerate(tiles):
        sheet.paste(im, ((i % SHEET_COLS) * (TILE_W + 6), (i // SHEET_COLS) * (th + 6)))
    return sheet


# ---------------------------------------------------------------- VLM call

def jpeg_b64(img: Image.Image, quality: int = 88) -> str:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def units(shot: dict) -> list[dict]:
    """A shot, or its parts if it is longer than SEGMENT_MAX_S (each part analysed on its own)."""
    n = max(1, math.ceil(shot["duration"] / SEGMENT_MAX_S - 1e-6))
    if n == 1:
        return [shot | {"key": f"shot_{shot['index']:02d}", "part": None}]
    out, step = [], shot["duration"] / n
    for k in range(n):
        a, b = shot["start"] + k * step, shot["start"] + (k + 1) * step
        out.append(shot | {
            "key": f"shot_{shot['index']:02d}_{k + 1}", "part": f"{k + 1}/{n}",
            "start": round(a, 3), "end": round(b, 3), "duration": round(b - a, 3),
            "cut_in": shot["cut_in"] if k == 0 else None,
            "words": [w for w in shot["words"] if a <= w["t"] < b],
            "zoom_events": [z for z in shot["zoom_events"] if a <= z["time"] < b],
            "jump_cuts": [t for t in shot["jump_cuts"] if a <= t < b],
            "on_screen_text": [o for o in shot["on_screen_text"] if o["first_seen"] < b and o["last_seen"] + 0.25 > a],
        })
    return out


def shot_context(shot: dict) -> str:
    a = shot["start"]
    span = f"#{shot['index']}  {a:.2f}s - {shot['end']:.2f}s ({shot['duration']:.2f}s)"
    if shot.get("part"):
        span += (f"  - this is part {shot['part']} of a longer shot; it continues from / into the "
                 f"neighbouring parts, so things may already be on screen or still moving at the edges")
    ctx = {
        "shot": span,
        "said": " ".join(f"{w['w']}@{w['t'] - a:+.2f}" for w in shot["words"]) or "(silence)",
        "layout": shot["layout"],
        "cut_in": shot["cut_in"] and {"type": shot["cut_in"]["type"], "framing_change": shot["cut_in"]["framing_change"]},
        "camera_moves": [{k: v for k, v in c.items() if k in ("moves", "zoom_total_pct")} for c in shot["camera"]],
        "zoom_events": [{"at_s": round(z["time"] - a, 2), "direction": z["direction"], "scale_pct": z["scale_pct"],
                         "frames": z["frames"], "kind": z["kind"]} for z in shot["zoom_events"]],
        "jump_cuts_at_s": [round(t - a, 2) for t in shot["jump_cuts"]],
        "on_screen_text_ocr": [o["text"] for o in shot["on_screen_text"]][:20],
    }
    return json.dumps(ctx, ensure_ascii=False, indent=1)


def captions_note(meta: dict) -> str:
    c = meta.get("captions") or {}
    y, wpc = c.get("center_y_median"), c.get("words_per_caption")
    if not y:
        return "the spoken-word subtitles"
    words = max(wpc, key=lambda k: wpc[k]) if wpc else "?"
    return f"the spoken-word subtitles, usually {words} word(s) at a time, centred about {y:.0%} down the frame"


def call_vlm(sheet: Image.Image, key: Image.Image, context: str, n: int, api_key: str, captions: str) -> dict:
    body = {
        "model": MODEL,
        "max_tokens": 6000,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{jpeg_b64(sheet)}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{jpeg_b64(key)}"}},
            {"type": "text", "text": PROMPT.format(n=n, context=context, captions=captions)},
        ]}],
    }
    for attempt in range(4):
        try:
            req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST", headers={
                "Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                resp = json.load(r)
            break
        except (urllib.error.URLError, TimeoutError) as e:
            if isinstance(e, urllib.error.HTTPError) and 400 <= e.code < 500 and e.code != 429:
                body = e.read().decode(errors='replace')[:300]
                raise RuntimeError(f'gateway HTTP {e.code} (not retried): {body}') from None
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))
            print(f"    retry after {e}", flush=True)
    text = resp["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", text, re.S)
    try:
        out = json.loads(m.group(0)) if m else {"raw": text}
    except json.JSONDecodeError:
        out = {"raw": text}
    usage = resp.get("usage") or {}
    out["_usage"] = {"input": usage.get("prompt_tokens"), "output": usage.get("completion_tokens"),
                     "cost": usage.get("cost")}
    return out


# ---------------------------------------------------------------- main

def analyse(video: Path, api_key: str, force: bool, only: set[int] | None) -> dict:
    out_dir = analysis_dir_for(video)
    timeline = load_json(out_dir / "timeline.json")
    if not timeline:
        raise RuntimeError("no timeline.json - run build_timeline.py first")
    info = probe_video(video)
    fps = info["fps"]
    vdir = out_dir / "vlm"
    vdir.mkdir(exist_ok=True)
    shots = [u for s in timeline["shots"] if only is None or s["index"] in only for u in units(s)]
    todo = [s for s in shots if force or not (vdir / f"{s['key']}.json").exists()]

    if todo:
        energy = change_energy(video, info, timeline)
        n_frames = len(energy)
        plan = {}
        for s in todo:
            a = max(0, round(s["start"] * fps))
            b = min(n_frames - 1, round(s["end"] * fps) - 1)
            plan[s["key"]] = (pick_frames(a, b, energy), a + int(0.8 * (b - a)))
        want = {f for picks, key in plan.values() for f in picks + [key]}
        frames = {}
        for f, bgr in enumerate(iter_frames(video, KEY_W, round(KEY_W * info["height"] / info["width"]))):
            if f in want:
                frames[f] = bgr.copy()

        def work(s):
            picks, key_f = plan[s["key"]]
            sheet = make_sheet(frames, picks, fps, s["start"])
            key = Image.fromarray(frames[key_f][:, :, ::-1])
            sheet.save(vdir / f"{s['key'].replace('shot', 'sheet')}.jpg", quality=85)
            key.save(vdir / f"{s['key'].replace('shot', 'key')}.jpg", quality=88)
            res = call_vlm(sheet, key, shot_context(s), len(picks), api_key, captions_note(timeline["meta"]))
            res["_shot"] = {"index": s["index"], "part": s["part"], "start": s["start"], "end": s["end"],
                            "frames": [round(f / fps, 3) for f in picks], "keyframe": round(key_f / fps, 3)}
            save_json(vdir / f"{s['key']}.json", res)
            print(f"  {s['key']}  {res.get('summary', res.get('raw', ''))[:110]}", flush=True)
            return res

        with ThreadPoolExecutor(PARALLEL) as pool:
            list(pool.map(work, todo))

    results = [load_json(vdir / f"{u['key']}.json") for s in timeline["shots"] for u in units(s)
               if (vdir / f"{u['key']}.json").exists()]
    cost = sum((r.get("_usage") or {}).get("cost") or 0 for r in results)
    combined = {"source": rel(video), "model": MODEL, "cost_usd": round(cost, 4), "shots": results}
    save_json(vdir / "vlm.json", combined)
    return combined


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", help="Videos or folders (default: all of reference/)")
    parser.add_argument("--force", action="store_true", help="redo shots that already have results")
    parser.add_argument("--shots", default="", help="comma-separated shot indexes to (re)do")
    args = parser.parse_args()
    load_dotenv(ROOT.parent / ".env")
    load_dotenv(ROOT / ".env")
    key = os.environ.get("VERCEL_AI_GATEWAY_API_KEY")
    if not key:
        parser.error("VERCEL_AI_GATEWAY_API_KEY not set (.env)")
    only = {int(x) for x in args.shots.split(",") if x.strip()} or None
    for video in collect_videos(args.paths):
        print(f"[vlm] {video.name}", flush=True)
        res = analyse(video, key, args.force, only)
        print(f"  -> {rel(analysis_dir_for(video) / 'vlm' / 'vlm.json')}  {len(res['shots'])} shots, "
              f"${res['cost_usd']:.2f} so far for this video")
    return 0


if __name__ == "__main__":
    sys.exit(main())
