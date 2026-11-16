"""Ask a vision model (via the Vercel AI Gateway) which font the captions use.

Used by match_font.py --vlm. The model sees a sheet of real caption crops plus the measured
proportions, and names (a) the font it thinks it is, which may be commercial, and (b) the closest
Google Fonts, which match_font.py then scores by rendering. The VLM only proposes; rendering decides.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import urllib.request

from dotenv import load_dotenv
from PIL import Image

from common import ROOT

MODEL = "anthropic/claude-opus-5.5"
URL = "https://ai-gateway.vercel.sh/v1/chat/completions"

PROMPT = """These are on-screen captions cut from one creator's short-form videos (one word at a time).
Identify the typeface.

Measured from the pixels (ratios to the x-height): x-height/cap-height {x_to_cap}, ascender {ascender_to_x},
stroke thickness {stroke_to_x}, gap between letters {letter_gap_to_x}, word space {word_space_to_x}.

Reply with JSON only:
{{
  "guesses": [{{"name": "...", "weight": "e.g. Bold / ExtraBold / 800", "free_on_google_fonts": true/false, "confidence": 0-1}}],
  "google_fonts": ["up to 10 Google Fonts family names that look closest, best first"],
  "letter_spacing": "tight / normal / loose",
  "distinguishing_features": "short note on the letterforms you used (a, g, t, y, R, G ...)"
}}
Up to 5 guesses, best first. Exact Google Fonts family names in google_fonts."""


def contact_sheet(crops: list[dict], n: int = 16) -> bytes:
    picks, seen = [], set()
    for c in sorted(crops, key=lambda c: -len(set(c["text"].lower()))):
        if c["text"].lower() not in seen and len(c["text"]) >= 4:
            seen.add(c["text"].lower())
            picks.append(c)
        if len(picks) == n:
            break
    tiles = [Image.fromarray(c["crop"]) for c in picks]
    w = max(t.width for t in tiles)
    sheet = Image.new("RGB", (w * 2, sum(t.height for t in tiles[::2]) + 10), (128, 128, 128))
    y = [0, 0]
    for i, t in enumerate(tiles):
        col = i % 2
        sheet.paste(t, (col * w, y[col]))
        y[col] += t.height
    buf = io.BytesIO()
    sheet.save(buf, "PNG")
    return buf.getvalue()


def ask_vlm(crops: list[dict], measured: dict) -> dict:
    load_dotenv(ROOT.parent / ".env")
    load_dotenv(ROOT / ".env")
    key = os.environ.get("VERCEL_AI_GATEWAY_API_KEY")
    if not key:
        raise RuntimeError("VERCEL_AI_GATEWAY_API_KEY not set (.env)")
    img = base64.b64encode(contact_sheet(crops)).decode()
    body = {
        "model": MODEL,
        "max_tokens": 1500,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}},
            {"type": "text", "text": PROMPT.format(**{k: measured.get(k) for k in (
                "x_to_cap", "ascender_to_x", "stroke_to_x", "letter_gap_to_x", "word_space_to_x")})},
        ]}],
    }
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST", headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        resp = json.load(r)
    text = resp["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", text, re.S)
    out = json.loads(m.group(0)) if m else {"raw": text}
    out["model"] = MODEL
    out["usage"] = resp.get("usage")
    print(f"VLM ({MODEL}) guesses: " + ", ".join(f"{g.get('name')} {g.get('weight', '')}".strip()
                                               for g in out.get("guesses", [])))
    print(f"VLM closest Google Fonts: {', '.join(out.get('google_fonts', []))}")
    return out
