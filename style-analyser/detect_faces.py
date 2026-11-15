"""Track faces to measure layout, framing, zoom levels and punch-ins at cuts.

Usage:
    uv run detect_faces.py                              # every video under reference/
    uv run detect_faces.py reference/kallawaymarketing  # one creator
    uv run detect_faces.py video.mp4 --force

Run detect_shots.py first: faces are summarised per shot and compared across every cut
(and every jump-cut candidate). Without shots.json the whole video is treated as one shot.

Writes to analysis/<creator>/<name>/:
    faces.json       per-shot layout + framing, zoom levels, what changes at each cut,
                     lined up with the transcript (camera zooms/pans/shake: detect_motion.py)
    faces_raw.npz    per-frame face boxes (primary face + every face) for re-analysis
    faces_plot.png   face size / position over time with cuts marked, for checking by eye

All sizes and positions are fractions of the frame (0..1), so they compare across resolutions.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from common import (ROOT, analysis_dir_for, collect_videos, load_json, near_boundary, probe_video, rel,
                    save_json, word_context)

MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
MODEL_PATH = ROOT / "models" / "face_detection_yunet_2023mar.onnx"

DETECT_W = 540                 # frames are resized to this width for detection
SCORE_THRESHOLD = 0.6
FULL_FACE_MIN_WIDTH = 0.25     # face box wider than this (fraction of frame width) = full-screen talking head
MIN_FACE_PRESENCE = 0.3        # shot needs a face in this share of frames to count as having one
BOUNDARY_WINDOW = 4            # frames averaged either side of a cut
PUNCH_THRESHOLD = 0.06         # >= 6% face-size change across a cut = punch-in/out
REFRAME_THRESHOLD = 0.03       # >= 3% of frame position shift across a cut = reframe
LEVEL_GAP = 0.06               # talking-head framings more than 6% apart in face size = different zoom level


# ---------------------------------------------------------------- detection

def ensure_model() -> Path:
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"  downloading face model -> {rel(MODEL_PATH)}")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def iter_frames(video: Path, width: int, height: int):
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(video), "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{width}x{height}", "pipe:"],
        stdout=subprocess.PIPE,
    )
    size = width * height * 3
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(height, width, 3)
    finally:
        proc.stdout.close()
        proc.wait()


def detect(video: Path, info: dict) -> tuple[np.ndarray, np.ndarray]:
    """Returns (primary [N, 8], all_faces [M, 6]).
    primary columns: cx, cy, w, h, eye_y, roll_deg, score, face_count (NaN where no face).
    all_faces columns: frame, cx, cy, w, h, score."""
    w = DETECT_W
    h = int(round(w * info["height"] / info["width"] / 2)) * 2
    detector = cv2.FaceDetectorYN.create(str(ensure_model()), "", (w, h), SCORE_THRESHOLD, 0.3, 20)
    primary, everyone = [], []
    for idx, frame in enumerate(iter_frames(video, w, h)):
        _, faces = detector.detect(frame)
        row = [np.nan] * 7 + [0]
        if faces is not None and len(faces):
            faces = faces[faces[:, 14] >= SCORE_THRESHOLD]
        if faces is not None and len(faces):
            for f in faces:
                everyone.append([idx, (f[0] + f[2] / 2) / w, (f[1] + f[3] / 2) / h, f[2] / w, f[3] / h, f[14]])
            best = faces[np.argmax(faces[:, 2] * faces[:, 3])]
            x, y, bw, bh = best[:4]
            (rex, rey), (lex, ley) = best[4:6], best[6:8]  # right eye, left eye (image coords)
            roll = float(np.degrees(np.arctan2(ley - rey, lex - rex)))
            row = [(x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h, (rey + ley) / 2 / h, roll, best[14], len(faces)]
        primary.append(row)
    return np.array(primary, np.float32), np.array(everyone, np.float32).reshape(-1, 6)


# ---------------------------------------------------------------- analysis helpers

def role(width: float) -> str:
    if np.isnan(width):
        return "none"
    return "full" if width >= FULL_FACE_MIN_WIDTH else "small"


def median_face(primary: np.ndarray, frames: range, want_role: str | None = None) -> dict | None:
    rows = [primary[f] for f in frames if 0 <= f < len(primary) and not np.isnan(primary[f, 0])]
    if want_role:
        rows = [r for r in rows if role(r[2]) == want_role]
    if not rows:
        return None
    m = np.median(np.array(rows), axis=0)
    return {"cx": float(m[0]), "cy": float(m[1]), "w": float(m[2]), "h": float(m[3]), "eye_y": float(m[4])}


def shot_summary(primary: np.ndarray, s: int, e: int, fps: float) -> dict:
    seg = primary[s:e + 1]
    has = ~np.isnan(seg[:, 0])
    presence = float(has.mean()) if len(seg) else 0.0
    out: dict = {"face_presence": round(presence, 2), "max_faces": int(seg[:, 7].max()) if len(seg) else 0}
    if presence < MIN_FACE_PRESENCE:
        out["layout"] = "no_face"
        return out
    face = seg[has]
    med = np.median(face, axis=0)
    full = med[2] >= FULL_FACE_MIN_WIDTH
    out["layout"] = "talking_head" if full else "face_inset"
    out["face"] = {
        "center_x": round(float(med[0]), 3), "center_y": round(float(med[1]), 3),
        "width": round(float(med[2]), 3), "height": round(float(med[3]), 3),
        "eye_line_y": round(float(med[4]), 3),
        "top_y": round(float(med[1] - med[3] / 2), 3),
        "roll_deg": round(float(med[5]), 1),
    }
    # Slow zoom inside the shot: fit log(face height) against time.
    idx = np.nonzero(has)[0]
    if len(idx) >= max(8, int(fps)):
        t = idx / fps
        logh = np.log(face[:, 3])
        slope, intercept = np.polyfit(t, logh, 1)
        pred = slope * t + intercept
        ss_res = float(((logh - pred) ** 2).sum())
        ss_tot = float(((logh - logh.mean()) ** 2).sum()) or 1e-9
        out["face_scale_trend"] = {  # includes leaning in/out; see motion.json for real zooms
            "pct_per_second": round(float(np.expm1(slope)) * 100, 2),
            "total_pct": round(float(np.expm1(slope * (t[-1] - t[0]))) * 100, 1),
            "r2": round(1 - ss_res / ss_tot, 2),
        }
        # Shake: high-frequency wobble of the face centre (head motion included, so read as an upper bound).
        k = 9
        if len(face) > k:
            cx = face[:, 0]
            smooth = np.convolve(np.pad(cx, k // 2, mode="edge"), np.ones(k) / k, mode="valid")
            out["face_jitter_pct"] = round(float(np.std(cx - smooth)) * 100, 2)
    return out


def boundary_change(primary: np.ndarray, frame: int) -> dict:
    """Compare the main face just before and just after a cut."""
    before = range(frame - BOUNDARY_WINDOW, frame)
    after = range(frame, frame + BOUNDARY_WINDOW)
    b_any, a_any = median_face(primary, before), median_face(primary, after)
    rb = role(b_any["w"]) if b_any else "none"
    ra = role(a_any["w"]) if a_any else "none"
    out: dict = {"before": rb, "after": ra}
    if rb == "none" or ra == "none" or rb != ra:
        out["change"] = "layout_change" if rb != ra else "no_face"
        return out
    ratio = a_any["h"] / b_any["h"]
    shift = float(np.hypot(a_any["cx"] - b_any["cx"], a_any["cy"] - b_any["cy"]))
    if ratio >= 1 + PUNCH_THRESHOLD:
        change = "punch_in"
    elif ratio <= 1 / (1 + PUNCH_THRESHOLD):
        change = "punch_out"
    elif shift >= REFRAME_THRESHOLD:
        change = "reframe"
    else:
        change = "same_framing"
    out.update({"change": change, "scale_pct": round((ratio - 1) * 100, 1),
                "shift_x": round(a_any["cx"] - b_any["cx"], 3), "shift_y": round(a_any["cy"] - b_any["cy"], 3)})
    return out


def zoom_levels(shots: list[dict]) -> list[dict]:
    """Group talking-head shots by face size: each group is one framing / zoom level."""
    th = sorted((s["face"]["height"], s["index"]) for s in shots if s.get("layout") == "talking_head")
    if not th:
        return []
    groups = [[th[0]]]
    for h, i in th[1:]:
        if h / groups[-1][-1][0] - 1 > LEVEL_GAP:
            groups.append([])
        groups[-1].append((h, i))
    base = float(np.median([h for h, _ in groups[0]]))
    return [{"level": k, "face_height": round(float(np.median([h for h, _ in g])), 3),
             "zoom_pct": round(float(np.median([h for h, _ in g])) / base * 100),
             "shots": [i for _, i in g]} for k, g in enumerate(groups)]


# ---------------------------------------------------------------- plot

def plot(primary: np.ndarray, fps: float, shots: list[dict], cuts: list[float], jumps: list[float],
         out: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(len(primary)) / fps
    fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True)
    series = [(primary[:, 3], "face height (frac of frame)"), (primary[:, 1], "face centre y (0=top)"),
              (primary[:, 0], "face centre x")]
    colors = {"talking_head": "#d8ecff", "face_inset": "#dff5df", "no_face": "#eeeeee"}
    for ax, (y, label) in zip(axes, series):
        for s in shots:
            ax.axvspan(s["start"], s["end"], color=colors.get(s.get("layout"), "white"), lw=0)
        ax.plot(t, y, ".", ms=2, color="#1f4e79")
        for c in cuts:
            ax.axvline(c, color="#888", lw=0.8)
        for j in jumps:
            ax.axvline(j, color="#f39c12", lw=0.8, ls="--")
        ax.set_ylabel(label, fontsize=8)
        ax.grid(alpha=0.2)
    axes[1].invert_yaxis()
    axes[0].set_title(f"{title}   blue=talking head  green=face inset  grey=no face | "
                      f"grey line=cut  orange=jump-cut candidate", fontsize=9)
    axes[-1].set_xlabel("seconds")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- main

def analyse(video: Path) -> dict:
    info = probe_video(video)
    fps = info["fps"]
    out_dir = analysis_dir_for(video)
    transcript = load_json(out_dir / "transcript.json")
    words = transcript["words"] if transcript else []
    shots_data = load_json(out_dir / "shots.json")

    primary, everyone = detect(video, info)
    n = len(primary)

    if shots_data:
        raw_shots = shots_data["shots"]
        cut_frames = [tr["new_shot_frame"] for tr in shots_data["transitions"]]
        jump_frames = [j["frame"] for j in shots_data["jump_cut_candidates"]]
    else:
        raw_shots = [{"index": 0, "start_frame": 0, "end_frame": n - 1, "start": 0.0, "end": n / fps}]
        cut_frames, jump_frames = [], []

    shots = []
    for s in raw_shots:
        summary = shot_summary(primary, s["start_frame"], min(s["end_frame"], n - 1), fps)
        shots.append({"index": s["index"], "start": s["start"], "end": s["end"], **summary})

    boundaries = []
    for kind, frames in (("cut", cut_frames), ("jump_cut_candidate", jump_frames)):
        for f in frames:
            t = f / fps
            boundaries.append({"time": round(t, 3), "frame": int(f), "source": kind,
                               **boundary_change(primary, f),
                               "speech": word_context(t, words), "near_text_boundary": near_boundary(t, words)})
    boundaries.sort(key=lambda b: b["time"])

    levels = zoom_levels(shots)
    level_of = {i: lv["zoom_pct"] for lv in levels for i in lv["shots"]}
    for s in shots:
        if s["index"] in level_of:
            s["zoom_pct"] = level_of[s["index"]]

    np.savez_compressed(out_dir / "faces_raw.npz", primary=primary, all_faces=everyone,
                        primary_columns=np.array(["cx", "cy", "w", "h", "eye_y", "roll_deg", "score", "face_count"]),
                        all_columns=np.array(["frame", "cx", "cy", "w", "h", "score"]))
    plot(primary, fps, shots, [f / fps for f in cut_frames], [f / fps for f in jump_frames],
         out_dir / "faces_plot.png", video.stem)

    # stats
    dur = {k: 0.0 for k in ("talking_head", "face_inset", "no_face")}
    for s in shots:
        dur[s["layout"]] += s["end"] - s["start"]
    total = sum(dur.values()) or 1
    changes: dict[str, int] = {}
    for b in boundaries:
        changes[b["change"]] = changes.get(b["change"], 0) + 1
    th = [s for s in shots if s["layout"] == "talking_head"]
    inset = [s for s in shots if s["layout"] == "face_inset"]

    def med(items, *keys):
        vals = [s["face"][k] for s in items for k in keys]
        return round(float(np.median(vals)), 3) if vals else None

    return {
        "source": rel(video),
        "detector": "YuNet (OpenCV) 2023mar",
        "frames": n,
        "stats": {
            "layout_share": {k: round(v / total, 2) for k, v in dur.items()},
            "boundary_changes": changes,
            "zoom_levels_pct": [lv["zoom_pct"] for lv in levels],
            "talking_head_framing": {
                "face_height": med(th, "height"), "face_center_x": med(th, "center_x"),
                "eye_line_y": med(th, "eye_line_y"), "face_top_y": med(th, "top_y"),
            } if th else None,
            "inset_framing": {
                "face_height": med(inset, "height"), "face_center_x": med(inset, "center_x"),
                "face_center_y": med(inset, "center_y"),
            } if inset else None,
            "median_face_jitter_pct": round(float(np.median([s["face_jitter_pct"] for s in shots if "face_jitter_pct" in s])), 2)
            if any("face_jitter_pct" in s for s in shots) else None,
        },
        "zoom_levels": levels,
        "boundaries": boundaries,
        "shots": shots,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", help="Videos or folders (default: all of reference/)")
    parser.add_argument("--force", action="store_true", help="Re-run even if faces.json exists")
    args = parser.parse_args()

    videos = collect_videos(args.paths)
    if not videos:
        parser.error("no videos found")
    for video in videos:
        out_file = analysis_dir_for(video) / "faces.json"
        if out_file.exists() and not args.force:
            print(f"[skip] {video.name} (already analysed)")
            continue
        if not (analysis_dir_for(video) / "shots.json").exists():
            print(f"  note: no shots.json for {video.name}; run detect_shots.py first for per-shot results")
        print(f"[faces] {video.name}", flush=True)
        result = analyse(video)
        save_json(out_file, result)
        s = result["stats"]
        print(f"  -> {rel(out_file)}  layouts {s['layout_share']}, cuts {s['boundary_changes']}, "
              f"zoom levels {s['zoom_levels_pct']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
