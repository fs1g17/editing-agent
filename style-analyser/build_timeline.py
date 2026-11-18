"""Merge every analysis step into one timeline per video (input for the VLM pass and the tastebook).

Usage:
    uv run build_timeline.py                       # every analysed video
    uv run build_timeline.py reference/kallawaymarketing/2026-11-08_DffZRmfMWrf.mp4

Reads transcript.json, shots.json, faces.json, motion.json, captions.json, colour.json (whichever
exist) and writes analysis/<creator>/<name>/timeline.json:
    meta     duration, fps, size, cut rate, layout shares, caption style, grade summary
    shots    per shot: time span, what is said, layout + framing, camera moves, how it was cut in,
             captions and other on-screen text visible during it, footage grade
    events   one time-sorted list: cuts, jump cuts, zoom events, layout changes, on-screen text
             appearing, sentence ends
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import analysis_dir_for, collect_videos, load_json, probe_video, rel, save_json


def overlaps(a0: float, a1: float, b0: float, b1: float) -> bool:
    return a0 < b1 and b0 < a1


def layout_share(shots: list[dict]) -> dict:
    total = sum(s["duration"] for s in shots) or 1
    out: dict[str, float] = {}
    for s in shots:
        out[s["layout"] or "unknown"] = out.get(s["layout"] or "unknown", 0) + s["duration"]
    return {k: round(v / total, 2) for k, v in out.items()}


def build(video: Path) -> dict:
    d = analysis_dir_for(video)
    info = probe_video(video)
    tr = load_json(d / "transcript.json") or {}
    sh = load_json(d / "shots.json") or {}
    fa = load_json(d / "faces.json") or {}
    mo = load_json(d / "motion.json") or {}
    ca = load_json(d / "captions.json") or {}
    co = load_json(d / "colour.json") or {}
    words = tr.get("words", [])

    face_shots = {s["index"]: s for s in fa.get("shots", [])}
    colour_shots = {s["index"]: s for s in co.get("shots", [])}
    boundaries = {round(b["time"], 2): b for b in fa.get("boundaries", [])}
    transitions = {round(t["time"], 2): t for t in sh.get("transitions", [])}
    segments = mo.get("segments", [])
    captions = ca.get("captions", [])
    other = ca.get("other_text", [])

    shots = []
    for s in sh.get("shots", []):
        a, b = s["start"], s["end"]
        f = face_shots.get(s["index"], {})
        t_in = transitions.get(round(a, 2))
        bnd = boundaries.get(round(a, 2))
        segs = [g for g in segments if g.get("shot_index") == s["index"]]
        caps = [c for c in captions if overlaps(c["start"], c["end"], a, b)]
        col = colour_shots.get(s["index"], {})
        shots.append({
            "index": s["index"], "start": a, "end": b, "duration": s["duration"],
            "said": s.get("text", ""),
            "words": [{"t": w["start"], "w": w["display"]} for w in words if a <= w["start"] < b],
            "cut_in": None if not t_in else {
                "type": t_in["type"], "speech": t_in.get("speech"), "near_text_boundary": t_in.get("near_text_boundary"),
                "framing_change": bnd.get("change") if bnd else None},
            # the colour step finds windows the face step can't (big face in a big window)
            "layout": "face_inset" if col.get("kind") == "face_inset" else f.get("layout"),
            "face": f.get("face"),
            "camera": [{"start": g["start"], "end": g["end"], "moves": g.get("moves"),
                        "zoom_total_pct": g.get("zoom_total_pct"), "shake_pct": g.get("shake_pct")} for g in segs],
            "zoom_events": [z for z in mo.get("zoom_events", []) if a <= z["time"] < b],
            "jump_cuts": [j["time"] for j in sh.get("jump_cut_candidates", []) if a <= j["time"] < b],
            "captions": {"count": len(caps), "texts": [c["text"] for c in caps][:40],
                         "emphasis": [c["text"] for c in caps if c.get("emphasis")]},
            "on_screen_text": [{"text": o["text"], "first_seen": o["first_seen"], "last_seen": o["last_seen"],
                                "box": o["box"]} for o in other if overlaps(o["first_seen"], o["last_seen"] + 0.25, a, b)],
            "frame_kind": col.get("kind"),
            "grade": {k: col["camera"].get(k) for k in ("exposure_L", "contrast_p95_p5", "chroma_mean")}
            if col.get("camera") else None,
        })

    events = []
    for t in sh.get("transitions", []):
        events.append({"t": t["time"], "type": f"cut ({t['type']})"})
    for j in sh.get("jump_cut_candidates", []):
        events.append({"t": j["time"], "type": "jump cut"})
    for z in mo.get("zoom_events", []):
        events.append({"t": z["time"], "type": f"zoom {z['direction']} {z['scale_pct']}% ({z['kind']}, {z['frames']} frames)"})
    for b in fa.get("boundaries", []):
        if b.get("change") in ("layout_change", "punch_in", "punch_out", "reframe"):
            events.append({"t": b["time"], "type": f"{b['change']} ({b.get('before')} -> {b.get('after')})"})
    for o in other:
        events.append({"t": o["first_seen"], "type": f"text appears: {o['text']}"})
    for w in words:
        if w.get("sentence_end"):
            events.append({"t": w["end"], "type": "sentence end"})
    events.sort(key=lambda e: e["t"])

    st = sh.get("stats", {})
    return {
        "source": rel(video),
        "meta": {
            "duration": info["duration"], "fps": info["fps"], "size": [info["width"], info["height"]],
            "words_per_minute": tr.get("stats", {}).get("words_per_minute"),
            "shot_count": st.get("shot_count"), "cuts_per_minute": st.get("cuts_per_minute"),
            "shot_duration": st.get("shot_duration"),
            "layout_share": layout_share(shots) or fa.get("stats", {}).get("layout_share"),
            "inset_window": co.get("inset_window") or co.get("split_window"),
            "camera_moves": mo.get("stats", {}).get("segment_moves"),
            "captions": {k: ca.get("stats", {}).get(k) for k in (
                "words_per_caption", "duration_median_s", "lead_median_s", "line_height", "center_y_median",
                "case", "light_text_share", "entry_animation")},
            "grade": co.get("camera_grade"), "look": co.get("look"), "skin": co.get("skin"),
            "graphics_palette": co.get("graphics"),
        },
        "shots": shots,
        "events": events,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", help="Videos or folders (default: all of reference/)")
    args = parser.parse_args()
    for video in collect_videos(args.paths):
        if not (analysis_dir_for(video) / "shots.json").exists():
            print(f"[skip] {video.name} (no shots.json)")
            continue
        tl = build(video)
        out = analysis_dir_for(video) / "timeline.json"
        save_json(out, tl)
        print(f"-> {rel(out)}  {len(tl['shots'])} shots, {len(tl['events'])} events")
    return 0


if __name__ == "__main__":
    sys.exit(main())
