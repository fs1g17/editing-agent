"""Download Google Fonts (OFL/Apache, free to ship) and precompute cheap stats per font instance.

Usage:
    uv run font_index.py                       # sans-serif + display families
    uv run font_index.py --categories SANS_SERIF,DISPLAY,SERIF

The google/fonts repo is cloned without file contents ("blobless"), then only the upright font
files of the chosen categories are checked out into models/google-fonts/ (~0.5-1 GB).
Variable fonts are sampled every 100 units of weight, so "Inter" becomes Inter 100 ... Inter 900.

Writes models/font_index.json: one entry per instance with
    xh       x-height in px at the 160 px/em render size
    cap      cap height in px
    stroke   average stroke thickness / x-height (how heavy it is)
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from math import ceil

import numpy as np

from common import save_json
from fontlib import FONTS_DIR, INDEX_PATH, FontInstance, font_axes, ink_box, stroke_width

REPO = "https://github.com/google/fonts.git"


def git(*args: str) -> None:
    subprocess.run(["git", "-C", str(FONTS_DIR), *args], check=True)


def read_metadata() -> list[dict]:
    families = []
    for meta in sorted(FONTS_DIR.glob("*/*/METADATA.pb")):
        txt = meta.read_text(encoding="utf-8")
        if 'subsets: "latin"' not in txt:
            continue
        name = re.search(r'^name: "(.+)"', txt, re.M)
        cats = re.findall(r'^category: "(\w+)"', txt, re.M)
        files = []
        for block in re.findall(r"fonts \{(.*?)\n\}", txt, re.S):
            style = re.search(r'style: "(\w+)"', block).group(1)
            weight = int(re.search(r"weight: (\d+)", block).group(1))
            fname = re.search(r'filename: "(.+?)"', block).group(1)
            if style == "normal" and all(f["file"] != fname for f in files):
                files.append({"file": fname, "weight": weight})
        if name and files:
            families.append({"family": name.group(1), "categories": cats,
                             "dir": meta.parent.relative_to(FONTS_DIR).as_posix(), "files": files})
    return families


def sync(categories: set[str]) -> list[dict]:
    if not (FONTS_DIR / ".git").exists():
        FONTS_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--no-checkout", "--sparse",
                        REPO, str(FONTS_DIR)], check=True)
        git("sparse-checkout", "set", "--no-cone", "/*/*/METADATA.pb")
        git("checkout", "main")
    families = [f for f in read_metadata() if categories & set(f["categories"])]
    # variable fonts are named like "Inter[opsz,wght].ttf": escape the brackets, which are glob syntax
    esc = lambda s: re.sub(r"([\[\]*?\\])", r"\\\1", s)
    patterns = ["/*/*/METADATA.pb"] + sorted({f"/{f['dir']}/{esc(x['file'])}" for f in families for x in f["files"]})
    print(f"checking out {len(patterns) - 1} font files from {len(families)} families ...", flush=True)
    t0 = time.time()
    proc = subprocess.run(["git", "-C", str(FONTS_DIR), "sparse-checkout", "set", "--no-cone", "--stdin"],
                          input="\n".join(patterns), text=True)
    if proc.returncode:
        raise RuntimeError("git sparse-checkout failed")
    print(f"  done in {time.time() - t0:.0f}s")
    return families


def instance_stats(inst: FontInstance) -> dict | None:
    out = {}
    for key, probe in (("xh", "xz"), ("cap", "HE")):
        cov = inst.render(probe)
        if cov is None or (box := ink_box(cov)) is None:
            return None
        out[key] = box[3] - box[1]
    cov = inst.render("nmu")
    if cov is None or out["xh"] < 10:
        return None
    out["stroke"] = round(stroke_width(cov >= 0.5) / out["xh"], 4)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--categories", default="SANS_SERIF,DISPLAY")
    args = parser.parse_args()
    families = sync(set(args.categories.split(",")))

    entries, t0 = [], time.time()
    for n, fam in enumerate(families):
        for spec in fam["files"]:
            path = FONTS_DIR / fam["dir"] / spec["file"]
            if not path.exists():
                continue
            try:
                axes = {tag: (lo, dflt, hi) for tag, lo, dflt, hi in font_axes(path)}
            except Exception:
                continue
            # Captions are big text: use the display optical size where a font has one
            # (Inter at opsz=32 is "Inter Display"), like CSS font-optical-sizing does at ~60px.
            base = {"opsz": float(min(axes["opsz"][2], 60))} if "opsz" in axes else {}
            if "wght" in axes:
                lo, _, hi = axes["wght"]
                weights = list(range(max(100, ceil(lo / 100) * 100), int(hi) + 1, 100)) or [int(axes["wght"][1])]
                variants = [(base | {"wght": float(w)}, w) for w in weights]
            else:
                variants = [(base, spec["weight"])]
            for var, weight in variants:
                try:
                    stats = instance_stats(FontInstance(path, var))
                except Exception as e:  # broken or unusual font files
                    print(f"  skip {path.name} {var}: {e}", file=sys.stderr)
                    continue
                if stats:
                    entries.append({"family": fam["family"], "category": fam["categories"][0], "weight": weight,
                                    "file": path.relative_to(FONTS_DIR).as_posix(), "axes": var, **stats})
        if n % 100 == 0:
            print(f"  {n}/{len(families)} families, {len(entries)} instances, {time.time() - t0:.0f}s", flush=True)
    save_json(INDEX_PATH, {"em": 160, "categories": args.categories.split(","), "instances": entries})
    strokes = np.array([e["stroke"] for e in entries])
    print(f"-> {INDEX_PATH}  {len(entries)} instances from {len(families)} families  "
          f"(stroke/x-height {strokes.min():.2f}..{strokes.max():.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
