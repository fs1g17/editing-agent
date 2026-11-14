"""Download reference videos (Instagram reels, TikToks, Shorts...) grouped by creator.

Usage:
    uv run download.py <url> [<url> ...]
    uv run download.py --file links.txt
    uv run download.py <url> --cookies-from-browser firefox   # if Instagram demands login
    uv run download.py <url> --proxy http://user:pass@host:port   # or set REFERENCE_PROXY

Each video lands in reference/<creator>/ as:
    <upload_date>_<id>.mp4         the video (best quality, merged to mp4)
    <upload_date>_<id>.info.json   metadata: caption, duration, fps, resolution, etc.

Already-downloaded videos are skipped (tracked in reference/.archive.txt).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import yt_dlp

ROOT = Path(__file__).resolve().parent
REFERENCE_DIR = ROOT / "reference"
ARCHIVE_FILE = REFERENCE_DIR / ".archive.txt"

# Instagram's extractor puts the @handle in `channel`; `uploader` is the display
# name and `uploader_id` is numeric. Other sites fill these differently, so fall
# back through them in order.
CREATOR_FIELDS = ("channel", "uploader_id", "uploader", "creator")


def creator_tag(info: dict) -> str:
    for field in CREATOR_FIELDS:
        value = info.get(field)
        if value:
            # Keep it filesystem-safe: handles are usually [a-z0-9._] already.
            return re.sub(r"[^\w.\-]", "_", str(value).lstrip("@")).strip("._") or "unknown"
    return "unknown"


def build_opts(args: argparse.Namespace) -> dict:
    opts: dict = {
        # Best video + best audio, merged into mp4 (ffmpeg required).
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "writeinfojson": True,
        "download_archive": str(ARCHIVE_FILE),
        "noplaylist": False,  # carousels/playlists download every video in them
        "retries": 5,
        "quiet": False,
        "no_warnings": False,
    }
    proxy = args.proxy or os.environ.get("REFERENCE_PROXY")
    if proxy:
        opts["proxy"] = proxy
    if args.cookies_from_browser:
        opts["cookiesfrombrowser"] = (args.cookies_from_browser,)
    if args.cookies:
        opts["cookiefile"] = args.cookies
    return opts


def download(url: str, opts: dict, creator_override: str | None) -> list[Path]:
    """Resolve the creator first, then download into reference/<creator>/."""
    probe_opts = {k: v for k, v in opts.items() if k != "download_archive"}
    with yt_dlp.YoutubeDL({**probe_opts, "quiet": True}) as probe:
        info = probe.extract_info(url, download=False)

    entries = info.get("entries") or [info]
    saved: list[Path] = []
    for entry in entries:
        if not entry:
            continue
        creator = creator_override or creator_tag(entry) or creator_tag(info)
        out_dir = REFERENCE_DIR / creator
        out_dir.mkdir(parents=True, exist_ok=True)
        entry_opts = {
            **opts,
            "outtmpl": str(out_dir / "%(upload_date>%Y-%m-%d|nodate)s_%(id)s.%(ext)s"),
        }
        with yt_dlp.YoutubeDL(entry_opts) as ydl:
            entry_url = entry.get("webpage_url") or entry.get("url") or url
            result = ydl.extract_info(entry_url, download=True)
            if result is None:  # skipped via archive
                continue
            path = Path(ydl.prepare_filename(result)).with_suffix(".mp4")
            saved.append(path)
            print(f"  -> {path.relative_to(ROOT)}  (@{creator})")
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("urls", nargs="*", help="Video/reel URLs")
    parser.add_argument("-f", "--file", help="Text file with one URL per line (# for comments)")
    parser.add_argument("--creator", help="Force the folder name instead of auto-detecting the handle")
    parser.add_argument("--proxy", help="Proxy URL (default: $REFERENCE_PROXY, else none)")
    parser.add_argument("--cookies-from-browser", metavar="BROWSER",
                        help="Use your logged-in browser session (firefox, chrome, edge...)")
    parser.add_argument("--cookies", metavar="FILE", help="Netscape-format cookies.txt")
    args = parser.parse_args()

    urls = list(args.urls)
    if args.file:
        for line in Path(args.file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    if not urls:
        parser.error("give at least one URL or --file")

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    opts = build_opts(args)

    failed: list[tuple[str, str]] = []
    for url in urls:
        print(f"[download] {url}", flush=True)
        try:
            download(url, opts, args.creator)
        except yt_dlp.utils.DownloadError as exc:
            msg = str(exc)
            if "login" in msg.lower() or "rate-limit" in msg.lower() or "cookies" in msg.lower():
                msg += ("\n  hint: Instagram wants a session - retry with "
                        "--cookies-from-browser firefox (or chrome/edge), or --proxy.")
            failed.append((url, msg))
            print(f"  !! failed: {msg}", file=sys.stderr)

    if failed:
        print(f"\n{len(failed)}/{len(urls)} failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
