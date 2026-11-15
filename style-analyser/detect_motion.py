"""Measure camera / frame motion: zooms, pans, shake, and the true punch-in at each jump cut.

Face size can't tell a digital zoom from the speaker leaning in. Background can: a zoom scales
everything, leaning in only grows the face. So this tracks feature points across the whole frame
(face masked out) and fits a scale + rotation + translation between consecutive frames.

Usage:
    uv run detect_motion.py reference/kallawaymarketing [--force]

Needs shots.json (detect_shots.py); uses faces_raw.npz (detect_faces.py) to mask the face if present.

Writes to analysis/<creator>/<name>/:
    motion.json       per-shot zoom / pan / shake, zoom events inside shots, punch-in % across cuts
    motion_raw.npz    per-frame scale, rotation, dx, dy, inlier count
    motion_plot.png   cumulative zoom and pan per shot, for checking by eye
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from common import (analysis_dir_for, collect_videos, load_json, near_boundary, probe_video, rel, save_json,
                    word_context)
from detect_faces import iter_frames

TRACK_W = 360                 # frames resized to this width for tracking
MIN_INLIERS = 20              # fewer inliers than this = transform not trusted for that frame
FACE_MASK_GROW = 1.4          # mask this much around the face box
ZOOM_WINDOW = 6               # frames over which an in-shot zoom is measured
ZOOM_THRESHOLD = 0.04         # >= 4% scale change within ZOOM_WINDOW = zoom event
PUSH_MIN_TOTAL = 0.05         # >= 5% steady scale change over a whole shot = slow push / pull
PAN_THRESHOLD = 0.05          # >= 5% of frame width/height of drift over a shot = pan


# ---------------------------------------------------------------- per-frame transforms

def face_mask(shape: tuple[int, int], face_row: np.ndarray | None) -> np.ndarray:
    mask = np.full(shape, 255, np.uint8)
    if face_row is not None and not np.isnan(face_row[0]):
        h, w = shape
        cx, cy, fw, fh = face_row[:4]
        gw, gh = fw * FACE_MASK_GROW / 2, fh * FACE_MASK_GROW / 2
        x0, x1 = int(max(0, (cx - gw) * w)), int(min(w, (cx + gw) * w))
        y0, y1 = int(max(0, (cy - gh) * h)), int(min(h, (cy + gh * 1.6) * h))  # extend down over the body
        mask[y0:y1, x0:x1] = 0
    return mask


def similarity(prev: np.ndarray, cur: np.ndarray, mask: np.ndarray) -> tuple[float, float, float, float, int]:
    """Scale, rotation (deg), dx, dy (fractions of frame size) mapping prev -> cur."""
    h, w = prev.shape
    pts = cv2.goodFeaturesToTrack(prev, maxCorners=400, qualityLevel=0.01, minDistance=7, mask=mask)
    if pts is None or len(pts) < MIN_INLIERS:
        return np.nan, np.nan, np.nan, np.nan, 0
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None, winSize=(21, 21), maxLevel=3)
    ok = status.ravel() == 1
    if ok.sum() < MIN_INLIERS:
        return np.nan, np.nan, np.nan, np.nan, int(ok.sum())
    M, inliers = cv2.estimateAffinePartial2D(pts[ok], nxt[ok], method=cv2.RANSAC, ransacReprojThreshold=1.0)
    n_in = int(inliers.sum()) if inliers is not None else 0
    if M is None or n_in < MIN_INLIERS:
        return np.nan, np.nan, np.nan, np.nan, n_in
    a, b = M[0, 0], M[1, 0]
    scale = float(np.hypot(a, b))
    rot = float(np.degrees(np.arctan2(b, a)))
    # translation of the frame centre (so a centred zoom reads as zero pan)
    c = np.array([w / 2, h / 2])
    moved = M[:, :2] @ c + M[:, 2]
    dx, dy = (moved - c) / np.array([w, h])
    return scale, rot, float(dx), float(dy), n_in


def match_across_cut(a: np.ndarray, b: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray) -> dict | None:
    """Jump cuts aren't continuous motion, so match features (ORB) instead of tracking them."""
    orb = cv2.ORB_create(1500)
    ka, da = orb.detectAndCompute(a, mask_a)
    kb, db = orb.detectAndCompute(b, mask_b)
    if da is None or db is None or len(ka) < MIN_INLIERS or len(kb) < MIN_INLIERS:
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    if len(matches) < MIN_INLIERS:
        return None
    pa = np.float32([ka[m.queryIdx].pt for m in matches])
    pb = np.float32([kb[m.trainIdx].pt for m in matches])
    M, inliers = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    n_in = int(inliers.sum()) if inliers is not None else 0
    if M is None or n_in < MIN_INLIERS:
        return None
    h, w = a.shape
    c = np.array([w / 2, h / 2])
    moved = M[:, :2] @ c + M[:, 2]
    dx, dy = (moved - c) / np.array([w, h])
    return {"scale_pct": round((float(np.hypot(M[0, 0], M[1, 0])) - 1) * 100, 1),
            "shift_x": round(float(dx), 3), "shift_y": round(float(dy), 3), "inliers": n_in}


# ---------------------------------------------------------------- analysis

def shot_motion(log_s: np.ndarray, dx: np.ndarray, dy: np.ndarray, s: int, e: int, fps: float) -> dict:
    seg = slice(s + 1, e + 1)  # transform i maps frame i-1 -> i, so skip the shot's first frame
    ls, x, y = log_s[seg], dx[seg], dy[seg]
    valid = ~np.isnan(ls)
    out: dict = {"tracked_share": round(float(valid.mean()), 2) if len(ls) else 0.0}
    if valid.sum() < max(4, int(fps / 3)):
        return out
    cum_scale = np.nancumsum(np.where(valid, ls, 0))
    cum_x = np.nancumsum(np.where(valid, x, 0))
    cum_y = np.nancumsum(np.where(valid, y, 0))
    total = float(np.expm1(cum_scale[-1]))
    out["zoom_total_pct"] = round(total * 100, 1)
    out["zoom_peak_pct"] = round(float(np.expm1(cum_scale[np.argmax(np.abs(cum_scale))])) * 100, 1)
    out["pan_x"] = round(float(cum_x[-1]), 3)
    out["pan_y"] = round(float(cum_y[-1]), 3)
    # shake: frame-to-frame translation minus its smoothed version
    k = 5
    if valid.sum() > k:
        xv, yv = x[valid], y[valid]
        sx = xv - np.convolve(np.pad(xv, k // 2, mode="edge"), np.ones(k) / k, mode="valid")
        sy = yv - np.convolve(np.pad(yv, k // 2, mode="edge"), np.ones(k) / k, mode="valid")
        out["shake_pct"] = round(float(np.sqrt((sx ** 2 + sy ** 2).mean())) * 100, 3)
    dur = (e - s + 1) / fps
    moves = []
    if abs(total) >= PUSH_MIN_TOTAL:
        moves.append(f"slow_{'push_in' if total > 0 else 'pull_out'}")
    if abs(cum_x[-1]) >= PAN_THRESHOLD:
        moves.append(f"pan_{'left' if cum_x[-1] > 0 else 'right'}")  # content moves right = camera pans left
    if abs(cum_y[-1]) >= PAN_THRESHOLD:
        moves.append(f"tilt_{'up' if cum_y[-1] > 0 else 'down'}")
    out["moves"] = moves or ["static"]
    out["zoom_rate_pct_per_s"] = round(total * 100 / dur, 2) if dur else None
    return out


def zoom_events(log_s: np.ndarray, starts: list[int], ends: list[int], fps: float) -> list[dict]:
    """Fast zooms inside shots: >= ZOOM_THRESHOLD over ZOOM_WINDOW frames."""
    thr = np.log1p(ZOOM_THRESHOLD)
    events = []
    for s, e in zip(starts, ends):
        ls = np.where(np.isnan(log_s[s + 1:e + 1]), 0, log_s[s + 1:e + 1])
        cum = np.r_[0, np.cumsum(ls)]
        i = 0
        while i + ZOOM_WINDOW < len(cum):
            d = cum[i + ZOOM_WINDOW] - cum[i]
            if abs(d) < thr:
                i += 1
                continue
            # extend while it keeps moving the same way
            j = i + ZOOM_WINDOW
            while j + 1 < len(cum) and np.sign(cum[j + 1] - cum[j]) == np.sign(d) and abs(cum[j + 1] - cum[j]) > thr / 10:
                j += 1
            k0 = i
            while k0 + 1 < j and abs(cum[k0 + 1] - cum[k0]) < thr / 10:  # trim flat lead-in
                k0 += 1
            steps = np.abs(np.diff(cum[k0:j + 1]))
            change = float(np.expm1(cum[j] - cum[k0]))
            abrupt = bool(len(steps) and steps.max() >= 0.7 * abs(cum[j] - cum[k0]))
            if abrupt and j >= len(cum) - 2:
                # a one-frame jump on the last frame of a segment is a cut boundary off by a frame
                i = j + 1
                continue
            f0 = s + k0
            events.append({"time": round(f0 / fps, 3), "end": round((s + j) / fps, 3),
                           "frames": int(j - k0), "direction": "in" if change > 0 else "out",
                           "scale_pct": round(change * 100, 1), "kind": "step" if abrupt else "ramp"})
            i = j + 1
    return events


def plot(log_s, dx, dy, inl, fps, starts, ends, cuts, zooms, out: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(log_s)
    cum = np.full(n, np.nan)
    cx = np.full(n, np.nan)
    cy = np.full(n, np.nan)
    for s, e in zip(starts, ends):
        seg = slice(s, e + 1)
        ls = np.where(np.isnan(log_s[seg]), 0, log_s[seg]); ls[0] = 0
        x = np.where(np.isnan(dx[seg]), 0, dx[seg]); x[0] = 0
        y = np.where(np.isnan(dy[seg]), 0, dy[seg]); y[0] = 0
        cum[seg] = np.expm1(np.cumsum(ls)) * 100
        cx[seg] = np.cumsum(x) * 100
        cy[seg] = np.cumsum(y) * 100
    t = np.arange(n) / fps
    fig, axes = plt.subplots(3, 1, figsize=(16, 7), sharex=True)
    for ax, y, lab in [(axes[0], cum, "zoom within shot (%)"), (axes[1], cx, "pan x (% width)"),
                       (axes[2], cy, "pan y (% height)")]:
        ax.plot(t, y, lw=1, color="#1f4e79")
        for c in cuts:
            ax.axvline(c, color="#888", lw=0.7)
        for z in zooms:
            ax.axvspan(z["time"], z["end"], color="#e74c3c", alpha=0.25, lw=0)
        ax.set_ylabel(lab, fontsize=8)
        ax.grid(alpha=0.2)
    axes[0].set_title(f"{title}   grey=cut (curves reset per shot)  red=zoom event", fontsize=9)
    axes[-1].set_xlabel("seconds")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def analyse(video: Path) -> dict:
    info = probe_video(video)
    fps = info["fps"]
    out_dir = analysis_dir_for(video)
    shots_data = load_json(out_dir / "shots.json")
    if not shots_data:
        raise RuntimeError("no shots.json - run detect_shots.py first")
    faces_path = out_dir / "faces_raw.npz"
    faces = np.load(faces_path)["primary"] if faces_path.exists() else None
    faces_json = load_json(out_dir / "faces.json")
    transcript = load_json(out_dir / "transcript.json")
    words = transcript["words"] if transcript else []

    cut_frames = sorted(tr["new_shot_frame"] for tr in shots_data["transitions"])
    jump_frames = sorted(j["frame"] for j in shots_data["jump_cut_candidates"])
    boundary_frames = set(cut_frames) | set(jump_frames)
    keep = {f - 1 for f in boundary_frames} | boundary_frames  # frames needed for cross-cut matching

    w = TRACK_W
    h = int(round(w * info["height"] / info["width"] / 2)) * 2
    rows, kept, prev, prev_mask = [], {}, None, None
    for i, frame in enumerate(iter_frames(video, w, h)):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = face_mask(gray.shape, faces[i] if faces is not None and i < len(faces) else None)
        if prev is None or i in boundary_frames:
            rows.append((np.nan, np.nan, np.nan, np.nan, 0))
        else:
            # mask both the old and new face position
            rows.append(similarity(prev, gray, cv2.bitwise_and(mask, prev_mask)))
        if i in keep:
            kept[i] = (gray, mask)
        prev, prev_mask = gray, mask
    arr = np.array(rows, np.float64)
    scale, rot, dx, dy, inl = arr.T
    log_s = np.log(scale)
    n = len(arr)

    # shots split at cuts AND jump-cut candidates (both break the continuous motion)
    starts = [0] + sorted(boundary_frames)
    ends = [f - 1 for f in sorted(boundary_frames)] + [n - 1]
    segments = [(s, e) for s, e in zip(starts, ends) if e >= s]
    starts, ends = [s for s, _ in segments], [e for _, e in segments]

    zooms = zoom_events(log_s, starts, ends, fps)
    for z in zooms:
        z["speech"] = word_context(z["time"], words)
        z["near_text_boundary"] = near_boundary(z["time"], words)

    layout_at = {}
    if faces_json:
        for s in faces_json["shots"]:
            layout_at[s["index"]] = s["layout"]
    shot_index = {sh["start_frame"]: sh["index"] for sh in shots_data["shots"]}

    segments_out = []
    for s, e in segments:
        idx = max((sf for sf in shot_index if sf <= s), default=0)
        segments_out.append({
            "start": round(s / fps, 3), "end": round((e + 1) / fps, 3), "shot_index": shot_index.get(idx),
            "layout": layout_at.get(shot_index.get(idx)),
            **shot_motion(log_s, dx, dy, s, e, fps),
        })

    across = []
    for f in sorted(boundary_frames):
        if f - 1 not in kept or f not in kept:
            continue
        (ga, ma), (gb, mb) = kept[f - 1], kept[f]
        m = match_across_cut(ga, gb, ma, mb)
        t = f / fps
        entry = {"time": round(t, 3), "source": "cut" if f in cut_frames else "jump_cut_candidate",
                 "same_background": m is not None}
        if m:
            change = "punch_in" if m["scale_pct"] >= 4 else "punch_out" if m["scale_pct"] <= -4 else \
                "reframe" if np.hypot(m["shift_x"], m["shift_y"]) >= 0.03 else "same_framing"
            entry.update({"change": change, "measured_by": "background", **m})
        else:
            # Background too dark/plain to match: fall back to the face-size comparison.
            fb = next((b for b in (faces_json or {}).get("boundaries", []) if b["frame"] == f), None)
            if fb and fb.get("change") in ("punch_in", "punch_out", "reframe", "same_framing"):
                entry.update({"change": fb["change"], "measured_by": "face", "scale_pct": fb.get("scale_pct")})
        entry["speech"] = word_context(t, words)
        across.append(entry)

    np.savez_compressed(out_dir / "motion_raw.npz", scale=scale, rotation_deg=rot, dx=dx, dy=dy, inliers=inl)
    plot(log_s, dx, dy, inl, fps, starts, ends, [f / fps for f in sorted(boundary_frames)], zooms,
         out_dir / "motion_plot.png", video.stem)

    move_counts: dict[str, int] = {}
    for seg in segments_out:
        for mv in seg.get("moves", []):
            move_counts[mv] = move_counts.get(mv, 0) + 1
    change_counts: dict[str, int] = {}
    for a in across:
        if a.get("change"):
            key = f'{a["change"]} ({a["measured_by"]})'
            change_counts[key] = change_counts.get(key, 0) + 1
    shakes = [s["shake_pct"] for s in segments_out if "shake_pct" in s]
    by_layout = {}
    for lay in ("talking_head", "face_inset", "no_face"):
        vals = [s["shake_pct"] for s in segments_out if s.get("layout") == lay and "shake_pct" in s]
        if vals:
            by_layout[lay] = round(float(np.median(vals)), 3)

    return {
        "source": rel(video),
        "method": "LK optical flow + RANSAC similarity per frame (face masked); ORB matching across cuts",
        "stats": {
            "tracked_share": round(float((inl >= MIN_INLIERS).mean()), 2),
            "segment_moves": move_counts,
            "zoom_events": len(zooms),
            "zoom_event_kinds": {k: sum(1 for z in zooms if z["kind"] == k) for k in ("step", "ramp")},
            "cut_framing_changes": change_counts,
            "median_shake_pct": round(float(np.median(shakes)), 3) if shakes else None,
            "median_shake_pct_by_layout": by_layout,
        },
        "zoom_events": zooms,
        "across_cuts": across,
        "segments": segments_out,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", help="Videos or folders (default: all of reference/)")
    parser.add_argument("--force", action="store_true", help="Re-run even if motion.json exists")
    args = parser.parse_args()
    videos = collect_videos(args.paths)
    if not videos:
        parser.error("no videos found")
    for video in videos:
        out_file = analysis_dir_for(video) / "motion.json"
        if out_file.exists() and not args.force:
            print(f"[skip] {video.name} (already analysed)")
            continue
        print(f"[motion] {video.name}", flush=True)
        try:
            result = analyse(video)
        except RuntimeError as exc:
            print(f"  !! {exc}", file=sys.stderr)
            continue
        save_json(out_file, result)
        s = result["stats"]
        print(f"  -> {rel(out_file)}  tracked {s['tracked_share']:.0%}, moves {s['segment_moves']}, "
              f"zooms {s['zoom_event_kinds']}, across cuts {s["cut_framing_changes"]}, "
              f"shake {s['median_shake_pct_by_layout']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
