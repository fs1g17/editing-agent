"""Check that the faces the face step found are the creator, not people inside graphics.

Usage:
    uv run verify_faces.py thevarunmayya

Faceless explainer creators show lots of other faces: a photo avatar in a motion graphic, a news
screenshot, b-roll of strangers. The face step (detect_faces.py) can't tell those from the
creator's own face window, so they inflate "face_inset" / "talking_head" and get graded as if they
were the creator's footage.

This step embeds one face per shot with OpenCV's SFace recogniser, groups shots by identity
across all of the creator's videos, and takes as the creator the identity seen in the most
videos (ties: most talking-head time, then most time overall). Shots whose face is someone else
are relabelled no_face in faces.json (the original label is kept as layout_detected), and the
layout shares are recomputed. Run it after detect_faces.py and before detect_colour.py.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from common import ANALYSIS_DIR, REFERENCE_DIR, ROOT, load_json, probe_video, rel, save_json
from detect_faces import SCORE_THRESHOLD, ensure_model

SFACE_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/"
             "face_recognition_sface_2021dec.onnx")
SFACE_PATH = ROOT / "models" / "face_recognition_sface_2021dec.onnx"
SAME_PERSON = 0.36        # SFace cosine similarity above this = same person (OpenCV's suggested 0.363)
SAMPLES_PER_SHOT = 3


def ensure_sface() -> Path:
    if not SFACE_PATH.exists():
        print(f"  downloading face recogniser -> {rel(SFACE_PATH)}")
        urllib.request.urlretrieve(SFACE_URL, SFACE_PATH)
    return SFACE_PATH


def grab(video: Path, t: float, w: int, h: int) -> np.ndarray | None:
    out = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:"], capture_output=True).stdout
    return np.frombuffer(out, np.uint8).reshape(h, w, 3).copy() if len(out) == w * h * 3 else None


def shot_embedding(video: Path, info: dict, shot: dict, primary: np.ndarray, det, rec) -> np.ndarray | None:
    """Mean normalised SFace feature of the shot's primary face over a few frames."""
    W, H, fps = info["width"], info["height"], info["fps"]
    feats = []
    for q in np.linspace(0.2, 0.8, SAMPLES_PER_SHOT):
        t = shot["start"] + q * (shot["end"] - shot["start"])
        f = min(len(primary) - 1, int(t * fps))
        if not np.isfinite(primary[f][0]):
            continue
        img = grab(video, t, W, H)
        if img is None:
            continue
        det.setInputSize((W, H))
        _, faces = det.detect(img)
        if faces is None or not len(faces):
            continue
        # the face the face step tracked: nearest to its stored centre
        cx, cy = primary[f][0] * W, primary[f][1] * H
        face = min(faces, key=lambda r: (r[0] + r[2] / 2 - cx) ** 2 + (r[1] + r[3] / 2 - cy) ** 2)
        feat = rec.feature(rec.alignCrop(img, face)).ravel()
        feats.append(feat / (np.linalg.norm(feat) + 1e-9))
    if not feats:
        return None
    m = np.mean(feats, axis=0)
    return m / (np.linalg.norm(m) + 1e-9)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("creator")
    args = parser.parse_args()
    creator = Path(args.creator).name
    det = cv2.FaceDetectorYN.create(str(ensure_model()), "", (320, 320), SCORE_THRESHOLD, 0.3, 20)
    rec = cv2.FaceRecognizerSF.create(str(ensure_sface()), "")

    items = []  # (video_name, shot, embedding)
    data = {}
    for video in sorted((REFERENCE_DIR / creator).glob("*.mp4")):
        out = ANALYSIS_DIR / creator / video.stem
        faces = load_json(out / "faces.json")
        if not faces or not (out / "faces_raw.npz").exists():
            continue
        info = probe_video(video)
        primary = np.load(out / "faces_raw.npz")["primary"]
        data[video.stem] = (out, faces)
        for s in faces["shots"]:
            layout = s.get("layout_detected", s["layout"])
            if layout in ("talking_head", "face_inset"):
                emb = shot_embedding(video, info, s, primary, det, rec)
                if emb is not None:
                    items.append((video.stem, s, emb))
    if not items:
        print("no faces to verify")
        return 0

    # greedy identity clusters
    clusters: list[dict] = []
    for vid, s, emb in items:
        best = max(clusters, key=lambda c: float(c["centre"] @ emb), default=None)
        if best is not None and float(best["centre"] @ emb) > SAME_PERSON:
            best["members"].append((vid, s, emb))
            c = np.mean([m[2] for m in best["members"]], axis=0)
            best["centre"] = c / (np.linalg.norm(c) + 1e-9)
        else:
            clusters.append({"centre": emb, "members": [(vid, s, emb)]})

    def rank(c):
        videos = {m[0] for m in c["members"]}
        th = sum(m[1]["end"] - m[1]["start"] for m in c["members"]
                 if m[1].get("layout_detected", m[1]["layout"]) == "talking_head")
        total = sum(m[1]["end"] - m[1]["start"] for m in c["members"])
        return (len(videos), th, total)
    clusters.sort(key=rank, reverse=True)
    creator_c = clusters[0]
    print(f"{len(items)} shots with faces, {len(clusters)} identities; creator = identity seen in "
          f"{rank(creator_c)[0]}/{len(data)} videos, {rank(creator_c)[2]:.1f}s on screen")

    is_creator = {(m[0], m[1]["index"]) for m in creator_c["members"]}
    sims = {(vid, s["index"]): float(creator_c["centre"] @ emb) for vid, s, emb in items}
    for vid, (out, faces) in data.items():
        changed = 0
        for s in faces["shots"]:
            s.setdefault("layout_detected", s["layout"])
            if s["layout_detected"] not in ("talking_head", "face_inset"):
                continue
            key = (vid, s["index"])
            ok = key in is_creator
            s["face_identity"] = {"is_creator": ok, "similarity_to_creator": round(sims.get(key, 0.0), 3)}
            if key not in sims:
                s["face_identity"]["is_creator"] = None  # face too small/unclear to check: keep label
                s["layout"] = s["layout_detected"]
                continue
            s["layout"] = s["layout_detected"] if ok else "no_face"
            if not ok:
                s["layout_note"] = "face found is not the creator (a photo or another person in the graphics)"
                changed += 1
        total = sum(s["end"] - s["start"] for s in faces["shots"]) or 1
        share = {}
        for s in faces["shots"]:
            share[s["layout"]] = share.get(s["layout"], 0) + (s["end"] - s["start"]) / total
        faces["stats"]["layout_share_detected"] = faces["stats"].get("layout_share_detected", faces["stats"]["layout_share"])
        faces["stats"]["layout_share"] = {k: round(share.get(k, 0.0), 2) for k in ("talking_head", "face_inset", "no_face")}
        faces["stats"]["identity_check"] = {"model": "SFace", "threshold": SAME_PERSON,
                                            "shots_relabelled_no_face": changed}
        save_json(out / "faces.json", faces)
        if changed and (out / "colour.json").exists():
            (out / "colour.json").unlink()  # graded with the wrong layouts: recompute
        print(f"  {vid}: {changed} shots relabelled no_face; layout {faces['stats']['layout_share']} "
              f"(was {faces['stats']['layout_share_detected']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
