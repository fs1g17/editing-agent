"""Turn a creator's analysed reference videos into TASTEBOOK.md: evidence-backed editing rules.

Usage:
    uv run build_tastebook.py kallawaymarketing
    uv run build_tastebook.py kallawaymarketing --facts-only      # just the measured facts, no LLM call

Approach (after "Download Your Editor's Brain"): repeated choices become specific rules with
source timestamps; measurements stay separate from interpretation; unknowns stay unknown; rules
describe what TRIGGERS a choice (a word, a topic change, a sentence end) rather than intervals.

    1. facts      code aggregates every measured number across the videos (cuts, layouts, zooms,
                  captions, font, grade, ...) with per-video values, so consistency is visible
    2. synthesis  an LLM reads the facts plus every shot's VLM description (graphics, animations,
                  icon style) and writes rules, each citing shots as <video>#<shot>@<time>s
    3. check      code verifies every cited shot exists and the time falls inside it; rules with
                  too little evidence are flagged

Writes analysis/<creator>/:
    tastebook_facts.json   step 1
    tastebook.json         step 2 + 3 (structured, for the editor agent)
    TASTEBOOK.md           readable version
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics as st
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from common import ANALYSIS_DIR, ROOT, load_json, rel, save_json

MODEL = "anthropic/claude-opus-5.5"
URL = "https://ai-gateway.vercel.sh/v1/chat/completions"


def vid_id(name: str) -> str:
    """'2026-11-08_DffZRmfMWrf' -> 'DffZRmfMWrf' (short id used in citations)."""
    return name.split("_", 1)[-1]


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(st.median(xs), 3) if xs else None


def rng(xs):
    xs = [x for x in xs if x is not None]
    return [round(min(xs), 3), round(max(xs), 3)] if xs else None


# ---------------------------------------------------------------- 1. measured facts

def font_note(font: dict) -> str:
    """Is the best free match the actual font or a lookalike? Judge against the calibration."""
    best, cal = font["best"]["score"], font.get("calibration") or {}
    same, others = cal.get("same_font"), list((cal.get("other_fonts") or {}).values())
    base = "overlap 1.0 = identical."
    if not same or not others:
        return base
    verdict = ("likely the actual font" if best >= same - 0.06 else
               "a lookalike, not the actual font (the real one is probably not on Google Fonts)")
    return (f"{base} The same font, faked through blur and compression, scores {same}; similar fonts score "
            f"{min(others)}-{max(others)} against each other. The best match scores {best}, so it is {verdict}. "
            f"VLM guesses for the real font are listed under vlm_guess.")

def facts(creator: str) -> dict:
    base = ANALYSIS_DIR / creator
    videos = sorted(p for p in base.iterdir() if (p / "timeline.json").exists())
    per_video, zooms, cut_speech, cut_boundary, layout_changes, cuts_total = {}, [], Counter(), Counter(), 0, 0
    shot_dur_by_layout: dict[str, list] = {}
    opening = {}
    for v in videos:
        vid = vid_id(v.name)
        tl = load_json(v / "timeline.json")
        sh = load_json(v / "shots.json") or {}
        mo = load_json(v / "motion.json") or {}
        fa = load_json(v / "faces.json") or {}
        tr = load_json(v / "transcript.json") or {}
        ca = load_json(v / "captions.json") or {}
        co = load_json(v / "colour.json") or {}
        m = tl["meta"]
        shots = tl["shots"]
        for s in shots:
            shot_dur_by_layout.setdefault(s["layout"] or "unknown", []).append(s["duration"])
        for a, b in zip(shots, shots[1:]):
            cuts_total += 1
            layout_changes += a["layout"] != b["layout"]
        for t in sh.get("transitions", []):
            cut_speech[(t.get("speech") or {}).get("position")] += 1
            cut_boundary[t.get("near_text_boundary") or "none"] += 1
        layout_at = lambda t: next((s["layout"] for s in shots if s["start"] <= t < s["end"]), None)
        for z in mo.get("zoom_events", []):
            zooms.append({"ref": f"{vid}@{z['time']:.2f}s", "layout": layout_at(z["time"]), "direction": z["direction"],
                          "scale_pct": z["scale_pct"], "frames": z["frames"], "kind": z["kind"],
                          "speech_position": (z.get("speech") or {}).get("position"),
                          "word": (z.get("speech") or {}).get("word") or (z.get("speech") or {}).get("next_word"),
                          "near_text_boundary": z.get("near_text_boundary")})
        opening[vid] = [{"shot": f"{vid}#{s['index']}", "t": [s["start"], s["end"]], "layout": s["layout"],
                         "said": s["said"]} for s in shots if s["start"] < 10]
        cst = ca.get("stats", {})
        g = co.get("camera_grade", {})
        per_video[vid] = {
            "file": v.name,
            "duration_s": m["duration"], "fps": m["fps"], "size": m["size"],
            "words_per_minute": (tr.get("stats") or {}).get("words_per_minute"),
            "pauses_over_0.3s": (tr.get("stats") or {}).get("pause_count"),
            "shot_count": m["shot_count"], "cuts_per_minute": m["cuts_per_minute"],
            "shot_duration": m["shot_duration"], "first_cut_at": (sh.get("stats") or {}).get("first_cut_at"),
            "transition_types": (sh.get("stats") or {}).get("transition_types"),
            "jump_cut_candidates": (sh.get("stats") or {}).get("jump_cut_candidates"),
            "layout_share": m["layout_share"],
            "talking_head_framing": (fa.get("stats") or {}).get("talking_head_framing"),
            "inset_framing": (fa.get("stats") or {}).get("inset_framing"),
            "inset_window": m.get("inset_window"),
            "zoom_levels_pct": (fa.get("stats") or {}).get("zoom_levels_pct"),
            "camera_moves_per_segment": (mo.get("stats") or {}).get("segment_moves"),
            "shake_pct_by_layout": (mo.get("stats") or {}).get("median_shake_pct_by_layout"),
            "cut_framing_changes": (mo.get("stats") or {}).get("cut_framing_changes"),
            "captions": {k: cst.get(k) for k in (
                "caption_count", "speech_coverage", "words_per_caption", "lines_per_caption", "duration_median_s",
                "lead_median_s", "linger_median_s", "line_height", "center_y_median", "center_x_median", "case",
                "punctuation_share", "light_text_share", "outline_share", "backing_box_share", "highlight_share",
                "emphasis_count", "entry_animation", "exit_animation")},
            "grade": {k: g.get(k) for k in ("exposure_L", "black_point_L", "white_point_L", "contrast_p95_p5",
                                            "chroma_mean", "colourfulness", "cast", "shadows_tint", "highlights_tint")},
            "look": co.get("look"), "skin": co.get("skin"),
            "graphics_canvas": (co.get("graphics") or {}).get("canvas"),
            "graphics_accents": [a["hex"] for a in (co.get("graphics") or {}).get("accents", [])],
        }
    font = load_json(base / "font" / "font.json")
    pv = per_video.values()
    return {
        "creator": creator,
        "videos": {k: v["file"] for k, v in per_video.items()},
        "summary": {
            "cuts_per_minute": {"median": med([v["cuts_per_minute"] for v in pv]), "range": rng([v["cuts_per_minute"] for v in pv])},
            "shot_duration_median_s": {"median": med([v["shot_duration"]["median"] for v in pv]),
                                       "range": rng([v["shot_duration"]["median"] for v in pv])},
            "shot_duration_by_layout_s": {k: {"median": med(x), "p10": round(sorted(x)[len(x) // 10], 2),
                                              "p90": round(sorted(x)[len(x) * 9 // 10], 2), "n": len(x)}
                                          for k, x in shot_dur_by_layout.items()},
            "cuts_that_change_layout": f"{layout_changes}/{cuts_total}",
            "cut_position_vs_speech": dict(cut_speech),
            "cut_near_text_boundary": dict(cut_boundary),
            "words_per_minute": {"median": med([v["words_per_minute"] for v in pv]), "range": rng([v["words_per_minute"] for v in pv])},
            "layout_share_face_inset": rng([(v["layout_share"] or {}).get("face_inset") for v in pv]),
            "zoom_events": {
                "count": len(zooms), "by_layout": dict(Counter(z["layout"] for z in zooms)),
                "direction": dict(Counter(z["direction"] for z in zooms)), "kind": dict(Counter(z["kind"] for z in zooms)),
                "scale_pct": {"median": med([abs(z["scale_pct"]) for z in zooms]), "range": rng([abs(z["scale_pct"]) for z in zooms])},
                "frames": {"median": med([z["frames"] for z in zooms]), "range": rng([z["frames"] for z in zooms])},
                "speech_position": dict(Counter(z["speech_position"] for z in zooms)),
                "near_text_boundary": dict(Counter(z["near_text_boundary"] or "none" for z in zooms)),
            },
        },
        "zoom_events": zooms,
        "opening_first_10s": opening,
        "per_video": per_video,
        "caption_font": None if not font else {
            "measured": {k: font["measured"].get(k) for k in ("x_height_px", "x_to_cap", "ascender_to_x", "stroke_to_x",
                                                               "letter_gap_to_x", "word_space_to_x")},
            "best_free_match": {k: font["best"].get(k) for k in ("family", "weight", "axes", "score", "size_px", "track_em")},
            "runners_up": [f"{f['family']} {f['weight']} ({f['score']})" for f in font["families"][1:5]],
            "calibration": font.get("calibration"),
            "vlm_guess": [g.get("name") for g in (font.get("vlm") or {}).get("guesses", [])],
            "note": font_note(font),
        },
        "sound": "not analysed yet (deferred by the user) - unknown",
    }


# ---------------------------------------------------------------- 2. synthesis

def compact_shots(creator: str) -> list[dict]:
    out = []
    for v in sorted((ANALYSIS_DIR / creator).iterdir()):
        vlm, tl = load_json(v / "vlm" / "vlm.json"), load_json(v / "timeline.json")
        if not vlm or not tl:
            continue
        vid = vid_id(v.name)
        by_idx = {s["index"]: s for s in tl["shots"]}
        for r in vlm["shots"]:
            i = r["_shot"]["index"]
            s = by_idx.get(i, {})
            a, b = r["_shot"].get("start", s.get("start", 0)), r["_shot"].get("end", s.get("end", 0))
            out.append({
                "ref": f"{vid}#{i}", "t": [round(a, 2), round(b, 2)],
                **({"part_of_shot": r["_shot"]["part"]} if r["_shot"].get("part") else {}),
                "layout": s.get("layout"),
                "said": " ".join(w["w"] for w in s.get("words", []) if a <= w["t"] < b) if r["_shot"].get("part")
                else s.get("said"),
                "zoom_events_at_s": [round(z["time"] - a, 2) for z in s.get("zoom_events", []) if a <= z["time"] < b],
                **{k: r.get(k) for k in ("summary", "visual_type", "background", "elements", "animations",
                                         "text_styles", "icon_style", "speech_link", "standouts", "raw") if r.get(k)},
            })
    return out


PROMPT = """You are writing TASTEBOOK.md for the short-form video creator @{creator}: the editing rules an
automated video editor will follow to reproduce their visual style on new raw footage (phone clips of a
person talking to camera). Sound is out of scope for now.

You get two kinds of input:
A) MEASURED FACTS - computed by code from the videos (cuts, layouts, zooms, captions, caption font,
   colour grade). These are reliable numbers. Copy numbers from here; never invent numbers.
B) SHOT DESCRIPTIONS - a vision model's description of every shot ({n_shots} shots, {n_videos} videos),
   from 12 sampled frames per shot: graphics, animations, icon styling, standouts. These are
   interpretations; timings in them are estimates from sampled frames. OCR'd text in them may contain
   misspellings - treat odd spellings as OCR noise, not as style.

Where A and B disagree, A wins (e.g. if code measured no caption animation, captions are not animated).

How to write rules:
- A rule is a repeated choice turned into an instruction an editor can execute. It needs evidence:
  at least 2 occurrences, ideally in more than one video. Count occurrences by scanning ALL shots.
- Describe TRIGGERS, not intervals: what in the speech/content causes the choice (a product name is said,
  a list starts, a sentence ends, a number is mentioned, a new topic begins), not "every 3 seconds".
- Be specific enough to build: sizes, positions, colours (hex), durations, easing, stroke styles.
- Cite evidence as "<video>#<shot>@<time>s", e.g. "DffZRmfMWrf#0@2.3s", using refs and times (seconds into
  the video, inside that shot's t range) from the input. Give 2-5 examples per rule, from different videos
  when possible, each with a few words on what happens there.
- basis: "measured" (from A), "observed" (from B), or "both".
- confidence: high = consistent across most videos; medium = clear but fewer cases; low = suggestive.
- Distinctive one-offs that are worth imitating but not repeated go in "signature_moments", not rules.
- Things the creator consistently does NOT do go in "avoid" (e.g. no crossfades), with basis.
- Anything that cannot be determined goes in "unknowns". Unknown stays unknown.
- Be concise: at most 8 rules per section (merge related ones), 2-4 examples per rule with a few words
  each, at most 15 animation vocabulary entries, 12 graphic templates, 10 signature moments.
- Normalise the animation vocabulary: different wordings of the same move ("blur-in fade", "fade/blur-in")
  become one canonical entry with its variants listed.

Reply with JSON only, this shape:
{{
  "summary": "4-6 sentences: the style in plain words, the way an editor would brief another editor",
  "sections": [
    {{"name": "Opening (first 10 s)", "rules": [RULE, ...]}},
    {{"name": "Format & layouts", "rules": [...]}},
    {{"name": "Pacing & cuts", "rules": [...]}},
    {{"name": "Camera & zoom", "rules": [...]}},
    {{"name": "Captions", "rules": [...]}},
    {{"name": "Typography (titles & on-screen text)", "rules": [...]}},
    {{"name": "Colour & grade", "rules": [...]}},
    {{"name": "Graphics & icon style", "rules": [...]}},
    {{"name": "Motion graphics & animation", "rules": [...]}},
    {{"name": "What to show when (speech -> visuals)", "rules": [...]}},
    {{"name": "B-roll & screen content", "rules": [...]}}
  ],
  "animation_vocabulary": [
    {{"name": "canonical name", "variants": ["wordings seen"], "what": "what moves and how",
      "duration_s": "typical range", "easing": "...", "used_for": "which elements / situations",
      "count": 0, "videos": 0, "examples": ["ref@t"]}}
  ],
  "graphic_templates": [
    {{"name": "e.g. 'outlined logo tile'", "spec": "exact look: shape, size, stroke, colours, effects",
      "used_for": "...", "count": 0, "examples": ["ref@t"]}}
  ],
  "signature_moments": [{{"ref": "ref@t", "what": "..."}}],
  "avoid": [{{"rule": "...", "basis": "measured|observed|both"}}],
  "unknowns": ["..."]
}}
where RULE = {{"id": "SECTION-N e.g. CUT-1", "rule": "imperative instruction", "trigger": "what causes it, or null",
  "parameters": {{"name": value}}, "basis": "measured|observed|both",
  "evidence": {{"videos": "k/{n_videos}", "count": 0, "examples": ["ref@t - what happens"]}},
  "confidence": "high|medium|low"}}

A) MEASURED FACTS
{facts}

B) SHOT DESCRIPTIONS
{shots}"""


def call_llm(prompt: str, api_key: str, raw_path: Path) -> tuple[dict, dict]:
    body = {"model": MODEL, "max_tokens": 64000, "messages": [{"role": "user", "content": prompt}]}
    for attempt in range(3):
        try:
            req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST", headers={
                "Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=1200) as r:
                resp = json.load(r)
            break
        except (urllib.error.URLError, TimeoutError) as e:
            if isinstance(e, urllib.error.HTTPError) and 400 <= e.code < 500 and e.code != 429:
                body = e.read().decode(errors='replace')[:300]
                raise RuntimeError(f'gateway HTTP {e.code} (not retried): {body}') from None
            if attempt == 2:
                raise
            print(f"  retry after {e}", flush=True)
            time.sleep(10)
    text = resp["choices"][0]["message"]["content"]
    raw_path.write_text(text, encoding="utf-8")  # keep the paid-for reply even if parsing fails
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)), resp.get("usage") or {}
    except (AttributeError, json.JSONDecodeError):
        raise RuntimeError(f"LLM reply was not JSON (finish: {resp['choices'][0].get('finish_reason')}):\n{text[-800:]}")


# ---------------------------------------------------------------- 3. check citations

REF = re.compile(r"([A-Za-z0-9_-]{6,})#(\d+)(?:@(\d+(?:\.\d+)?)s)?")


def check(tb: dict, shots: list[dict], n_videos: int) -> dict:
    spans: dict[str, list] = {}
    for s in shots:  # a long shot may come as several parts: its span is their union
        lo, hi = spans.get(s["ref"], s["t"])
        spans[s["ref"]] = [min(lo, s["t"][0]), max(hi, s["t"][1])]
    bad, total = [], 0

    def scan(text: str, where: str):
        nonlocal total
        for vid, idx, t in REF.findall(text):
            total += 1
            ref = f"{vid}#{idx}"
            if ref not in spans:
                bad.append(f"{where}: unknown shot {ref}")
            elif t and not (spans[ref][0] - 0.3 <= float(t) <= spans[ref][1] + 0.3):
                bad.append(f"{where}: {ref}@{t}s is outside the shot ({spans[ref][0]}-{spans[ref][1]}s)")

    weak = []
    for sec in tb.get("sections", []):
        for r in sec.get("rules", []):
            ex = (r.get("evidence") or {}).get("examples") or []
            for e in ex:
                scan(e, r.get("id", "?"))
            if len(ex) < 2 and r.get("basis") != "measured":
                weak.append(f"{r.get('id')}: fewer than 2 cited examples")
    for key in ("animation_vocabulary", "graphic_templates"):
        for item in tb.get(key, []):
            for e in item.get("examples", []):
                scan(e, f"{key}:{item.get('name')}")
    for sm in tb.get("signature_moments", []):
        scan(sm.get("ref", ""), "signature_moment")
    return {"citations_checked": total, "invalid": bad, "weak_rules": weak}


# ---------------------------------------------------------------- markdown

def to_markdown(tb: dict, f: dict, model: str) -> str:
    L = [f"# TASTEBOOK - @{f['creator']}", ""]
    L += [f"_Built from {len(f['videos'])} reference videos: " + ", ".join(f"`{k}`" for k in f["videos"]) +
          f". Measured by code (TransNetV2 cuts, YuNet faces, optical-flow camera motion, OCR captions, font "
          f"render-matching, Lab colour) and interpreted by {model} from sampled frames. Citations are "
          f"`<video>#<shot>@<seconds>`._", ""]
    L += ["## The style in brief", "", tb.get("summary", ""), ""]
    for sec in tb.get("sections", []):
        if not sec.get("rules"):
            continue
        L += [f"## {sec['name']}", ""]
        for r in sec["rules"]:
            ev = r.get("evidence") or {}
            L.append(f"**{r.get('id')}** - {r.get('rule')}  ")
            if r.get("trigger"):
                L.append(f"_Trigger:_ {r['trigger']}  ")
            if r.get("parameters"):
                L.append("_Parameters:_ " + "; ".join(f"{k} = {v}" for k, v in r["parameters"].items()) + "  ")
            L.append(f"_Basis:_ {r.get('basis')} · _confidence:_ {r.get('confidence')} · "
                     f"_seen in:_ {ev.get('videos', '?')} videos, {ev.get('count', '?')}×")
            for e in ev.get("examples", []):
                L.append(f"- {e}")
            L.append("")
    if tb.get("animation_vocabulary"):
        L += ["## Animation vocabulary", "", "| Move | What | Duration | Easing | Used for | Seen | Examples |",
              "|---|---|---|---|---|---|---|"]
        for a in tb["animation_vocabulary"]:
            L.append(f"| **{a.get('name')}** | {a.get('what')} | {a.get('duration_s')} | {a.get('easing')} | "
                     f"{a.get('used_for')} | {a.get('count')}× in {a.get('videos')} videos | "
                     f"{', '.join(a.get('examples', [])[:3])} |")
        L.append("")
    if tb.get("graphic_templates"):
        L += ["## Graphic templates", ""]
        for g in tb["graphic_templates"]:
            L += [f"**{g.get('name')}** ({g.get('count')}×) - {g.get('spec')}  ", f"_Used for:_ {g.get('used_for')}  ",
                  f"_Examples:_ {', '.join(g.get('examples', []))}", ""]
    if tb.get("signature_moments"):
        L += ["## Signature moments", ""] + [f"- `{s.get('ref')}` - {s.get('what')}" for s in tb["signature_moments"]] + [""]
    if tb.get("avoid"):
        L += ["## Avoid", ""] + [f"- {a.get('rule')} _({a.get('basis')})_" for a in tb["avoid"]] + [""]
    if tb.get("unknowns"):
        L += ["## Unknown", ""] + [f"- {u}" for u in tb["unknowns"]] + [""]
    v = tb.get("_validation", {})
    L += ["## Evidence check", "",
          f"{v.get('citations_checked', 0)} citations checked against the shot list; "
          f"{len(v.get('invalid', []))} invalid, {len(v.get('weak_rules', []))} rules with thin evidence."]
    L += [f"- {x}" for x in v.get("invalid", []) + v.get("weak_rules", [])]
    return "\n".join(L) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("creator")
    parser.add_argument("--facts-only", action="store_true")
    args = parser.parse_args()
    creator = Path(args.creator).name
    out = ANALYSIS_DIR / creator

    f = facts(creator)
    save_json(out / "tastebook_facts.json", f)
    print(f"-> {rel(out / 'tastebook_facts.json')}")
    if args.facts_only:
        return 0
    shots = compact_shots(creator)
    if not shots:
        parser.error('no VLM shot descriptions found - run vlm_shots.py first')
    load_dotenv(ROOT.parent / ".env")
    load_dotenv(ROOT / ".env")
    key = os.environ.get("VERCEL_AI_GATEWAY_API_KEY")
    if not key:
        parser.error("VERCEL_AI_GATEWAY_API_KEY not set (.env)")
    prompt = PROMPT.format(creator=creator, n_shots=len(shots), n_videos=len(f["videos"]),
                           facts=json.dumps(f, ensure_ascii=False), shots=json.dumps(shots, ensure_ascii=False))
    print(f"synthesising from {len(shots)} shots (~{len(prompt) // 4:,} tokens in) with {MODEL} ...", flush=True)
    t0 = time.time()
    tb, usage = call_llm(prompt, key, out / "tastebook_raw.txt")
    tb["_validation"] = check(tb, shots, len(f["videos"]))
    tb["_meta"] = {"model": MODEL, "seconds": round(time.time() - t0), "usage": {
        "input": usage.get("prompt_tokens"), "output": usage.get("completion_tokens"), "cost": usage.get("cost")}}
    save_json(out / "tastebook.json", tb)
    (out / "TASTEBOOK.md").write_text(to_markdown(tb, f, MODEL), encoding="utf-8")
    v = tb["_validation"]
    n_rules = sum(len(s.get("rules", [])) for s in tb.get("sections", []))
    print(f"-> {rel(out / 'tastebook.json')}\n-> {rel(out / 'TASTEBOOK.md')}\n"
          f"{n_rules} rules, {len(tb.get('animation_vocabulary', []))} animation moves, "
          f"{len(tb.get('graphic_templates', []))} graphic templates; citations {v['citations_checked']} checked, "
          f"{len(v['invalid'])} invalid; {tb['_meta']['seconds']}s, ${usage.get('cost') or 0:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
