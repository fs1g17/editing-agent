"""Run the whole style analysis for a creator, start to finish: download -> analyse -> VLM -> tastebook.

Usage:
    uv run pipeline.py https://www.instagram.com/p/AAA/ https://www.instagram.com/p/BBB/ ...
    uv run pipeline.py -f links.txt --vlm-jobs 5
    uv run pipeline.py --creator kallawaymarketing          # skip download, (re)run on what's there

Stages (every step skips work that already exists, so a rerun resumes where it stopped):
    1. download   download.py into reference/<creator>/
    2. analyse    per video, all videos in parallel: transcribe -> shots -> faces -> motion ->
                  captions -> colour
    3. font       match_font.py --vlm across the creator's captions
    4. vlm        per video, all videos in parallel: build_timeline -> vlm_shots
    5. tastebook  build_tastebook.py -> analysis/<creator>/TASTEBOOK.md

Each video's step output goes to analysis/<creator>/<video>/pipeline.log.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from common import REFERENCE_DIR, ROOT, analysis_dir_for, rel

ANALYSE_STEPS = ["transcribe.py", "detect_shots.py", "detect_faces.py", "detect_motion.py",
                 "detect_captions.py", "detect_colour.py"]
VLM_STEPS = ["build_timeline.py", "vlm_shots.py"]
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}


def run(args: list[str], log: Path | None = None) -> None:
    cmd = [sys.executable, *args]
    if log is None:
        subprocess.run(cmd, cwd=ROOT, env=ENV, check=True)
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(f"\n$ {' '.join(args)}\n")
        fh.flush()
        res = subprocess.run(cmd, cwd=ROOT, env=ENV, stdout=fh, stderr=subprocess.STDOUT)
    if res.returncode:
        raise RuntimeError(f"{args[0]} failed (see {rel(log)})")


def chain(video: Path, steps: list[str]) -> str:
    log = analysis_dir_for(video) / "pipeline.log"
    t0 = time.time()
    for step in steps:
        run([step, str(video)], log)
        print(f"  {video.stem}: {step} done ({time.time() - t0:.0f}s)", flush=True)
    return video.stem


def parallel(videos: list[Path], steps: list[str], jobs: int) -> list[str]:
    failed = []
    with ThreadPoolExecutor(jobs) as pool:
        futs = {pool.submit(chain, v, steps): v for v in videos}
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:
                failed.append(futs[f].stem)
                print(f"  !! {futs[f].stem}: {e}", flush=True)
    return failed


def video_ids(urls: list[str]) -> list[str]:
    ids = []
    for u in urls:
        m = re.search(r"/(?:p|reel|reels|tv)/([A-Za-z0-9_-]+)", u)
        if m:
            ids.append(m.group(1))
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("urls", nargs="*")
    parser.add_argument("-f", "--file", help="text file with one URL per line")
    parser.add_argument("--creator", help="creator folder to analyse (skips download if no URLs)")
    parser.add_argument("--analyse-jobs", type=int, default=1,
                        help="videos analysed at the same time (caption OCR is CPU-bound: 1 is fastest overall)")
    parser.add_argument("--vlm-jobs", type=int, default=5,
                        help="videos sent to the VLM at the same time (network-bound: parallel is fastest)")
    parser.add_argument("--no-tastebook", action="store_true")
    args = parser.parse_args()
    urls = list(args.urls)
    if args.file:
        urls += [l.strip() for l in Path(args.file).read_text(encoding="utf-8").splitlines()
                 if l.strip() and not l.startswith("#")]
    t0 = time.time()

    # 1. download
    if urls:
        print(f"[1/5 download] {len(urls)} links", flush=True)
        run(["download.py", *urls] + (["--creator", args.creator] if args.creator else []))
        ids = video_ids(urls)
        videos = sorted({p for i in ids for p in REFERENCE_DIR.glob(f"*/*_{i}.mp4")})
        missing = [i for i in ids if not any(v.stem.endswith(i) for v in videos)]
        if missing:
            print(f"  !! not downloaded: {missing}", flush=True)
    elif args.creator:
        videos = sorted((REFERENCE_DIR / args.creator).glob("*.mp4"))
    else:
        parser.error("give URLs or --creator")
    creators = sorted({v.parent.name for v in videos})
    if len(creators) != 1:
        parser.error(f"expected one creator, got {creators}")
    creator = creators[0]
    print(f"  @{creator}: {len(videos)} videos", flush=True)

    # 2. per-video analysis
    print(f"[2/5 analyse] {len(videos)} videos, {args.analyse_jobs} at a time", flush=True)
    failed = parallel(videos, ANALYSE_STEPS, args.analyse_jobs)
    videos = [v for v in videos if v.stem not in failed]
    if not videos:
        print("  !! every video failed in analysis - stopping (see the pipeline.log files)", flush=True)
        return 1

    # 3. caption font
    print("[3/5 font]", flush=True)
    if (ROOT / "analysis" / creator / "font" / "font.json").exists():
        print("  [skip] font.json exists", flush=True)
    else:
      try:
        run(["match_font.py", creator, "--vlm"], ROOT / "analysis" / creator / "font" / "pipeline.log")
      except RuntimeError as e:
        print(f"  !! {e} (continuing without a font match)", flush=True)

    # 4. VLM shot pass
    print(f"[4/5 vlm] {len(videos)} videos in parallel", flush=True)
    failed += parallel(videos, VLM_STEPS, args.vlm_jobs)

    # 5. tastebook
    if failed:
        print(f'  !! not building the tastebook: {len(failed)} video(s) failed (see their pipeline.log). '
              f'Fix the cause and rerun the same command - finished work is kept.', flush=True)
        return 1
    if not args.no_tastebook:
        print("[5/5 tastebook]", flush=True)
        run(["build_tastebook.py", creator])
    print(f"done in {(time.time() - t0) / 60:.1f} min" + (f"; failed: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
