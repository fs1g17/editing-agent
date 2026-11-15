"""Transcribe reference videos with word-level timestamps (Whisper via Vercel AI Gateway).

Usage:
    uv run transcribe.py                                   # every video under reference/
    uv run transcribe.py reference/kallawaymarketing       # one creator
    uv run transcribe.py path/to/video.mp4 --force         # redo an existing transcript

Needs VERCEL_AI_GATEWAY_API_KEY in the repo-root .env (or the environment).

For reference/<creator>/<name>.mp4 it writes analysis/<creator>/<name>/:
    audio.mp3         mono 16 kHz audio sent to Whisper (kept for later beat/energy analysis)
    transcript.json   full text, per-word timings with punctuation, pauses and pacing stats
"""

from __future__ import annotations

import argparse
import base64
import difflib
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

from common import ROOT, analysis_dir_for, collect_videos, rel

GATEWAY_URL = "https://ai-gateway.vercel.sh/v4/ai/transcription-model"
MODEL = "openai/whisper-1"
PAUSE_THRESHOLD = 0.3  # seconds of silence between words that counts as a pause
MAX_UPLOAD_MB = 24     # Whisper's own limit is 25 MB


def extract_audio(video: Path, out: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
         "-b:a", "64k", str(out)],
        check=True,
    )


def call_whisper(audio: Path, api_key: str) -> dict:
    size_mb = audio.stat().st_size / 1e6
    if size_mb > MAX_UPLOAD_MB:
        raise RuntimeError(f"{audio.name} is {size_mb:.1f} MB, over Whisper's 25 MB limit")
    body = {
        "audio": base64.b64encode(audio.read_bytes()).decode(),
        "mediaType": "audio/mpeg",
        "providerOptions": {"openai": {"timestampGranularities": ["word"]}},
    }
    req = urllib.request.Request(
        GATEWAY_URL,
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "ai-gateway-protocol-version": "0.0.1",
            "ai-transcription-model-specification-version": "4",
            "ai-model-id": MODEL,
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"gateway HTTP {exc.code}: {exc.read().decode(errors='replace')[:500]}") from None


def _norm(token: str) -> str:
    return re.sub(r"[^\w]", "", token.lower())


def attach_punctuation(words: list[dict], text: str) -> None:
    """Whisper's word list has no punctuation; the full text does. Align the two and copy
    each text token's spelling (e.g. "Claude," / "machine.") onto the matching word."""
    tokens = text.split()
    a = [_norm(w["word"]) for w in words]
    b = [_norm(t) for t in tokens]
    for block in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
        for k in range(block.size):
            words[block.a + k]["display"] = tokens[block.b + k]
    for w in words:
        w.setdefault("display", w["word"])
        w["sentence_end"] = w["display"].endswith((".", "?", "!"))
        w["clause_end"] = w["sentence_end"] or w["display"].endswith((",", ";", ":"))


def build_transcript(raw: dict, video: Path) -> dict:
    words = [
        {"word": s["text"].strip(), "start": round(s["startSecond"], 3), "end": round(s["endSecond"], 3)}
        for s in raw.get("segments") or []
    ]
    attach_punctuation(words, raw.get("text", ""))

    pauses = []
    for prev, nxt in zip(words, words[1:]):
        gap = nxt["start"] - prev["end"]
        if gap >= PAUSE_THRESHOLD:
            pauses.append({"after_word": prev["display"], "start": prev["end"], "end": nxt["start"],
                           "duration": round(gap, 3)})

    # Sentences: consecutive words up to a sentence-ending punctuation mark.
    sentences, current = [], []
    for w in words:
        current.append(w)
        if w["sentence_end"]:
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)

    speech_span = (words[-1]["end"] - words[0]["start"]) if words else 0
    duration = raw.get("durationInSeconds")
    return {
        "source": rel(video),
        "model": MODEL,
        "language": raw.get("language"),
        "duration": duration,
        "text": raw.get("text", ""),
        "stats": {
            "word_count": len(words),
            "words_per_minute": round(len(words) / speech_span * 60, 1) if speech_span else None,
            "first_word_at": words[0]["start"] if words else None,
            "last_word_at": words[-1]["end"] if words else None,
            "pause_count": len(pauses),
            "mean_pause": round(sum(p["duration"] for p in pauses) / len(pauses), 3) if pauses else 0,
            "sentence_count": len(sentences),
        },
        "sentences": [
            {"text": " ".join(w["display"] for w in s), "start": s[0]["start"], "end": s[-1]["end"]}
            for s in sentences
        ],
        "pauses": pauses,
        "words": words,
        "warnings": raw.get("warnings") or [],
        "cost_usd": _cost(raw),
    }


def _cost(raw: dict) -> float | None:
    try:
        return round(float(raw["providerMetadata"]["gateway"]["cost"]), 5)
    except (KeyError, TypeError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", help="Videos or folders (default: all of reference/)")
    parser.add_argument("--force", action="store_true", help="Re-transcribe even if transcript.json exists")
    args = parser.parse_args()

    load_dotenv(ROOT.parent / ".env")
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("VERCEL_AI_GATEWAY_API_KEY")
    if not api_key:
        parser.error("VERCEL_AI_GATEWAY_API_KEY is not set (put it in the repo-root .env)")

    videos = collect_videos(args.paths)
    if not videos:
        parser.error("no videos found")

    total_cost, failed = 0.0, 0
    for video in videos:
        out_dir = analysis_dir_for(video)
        out_file = out_dir / "transcript.json"
        if out_file.exists() and not args.force:
            print(f"[skip] {video.name} (already transcribed)")
            continue
        print(f"[transcribe] {video.name}", flush=True)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            audio = out_dir / "audio.mp3"
            extract_audio(video, audio)
            transcript = build_transcript(call_whisper(audio, api_key), video)
        except (RuntimeError, subprocess.CalledProcessError, urllib.error.URLError) as exc:
            failed += 1
            print(f"  !! failed: {exc}", file=sys.stderr)
            continue
        out_file.write_text(json.dumps(transcript, indent=2, ensure_ascii=False), encoding="utf-8")
        s = transcript["stats"]
        total_cost += transcript["cost_usd"] or 0
        print(f"  -> {out_file.relative_to(ROOT)}  {s['word_count']} words, {s['words_per_minute']} wpm, "
              f"{s['pause_count']} pauses, ${transcript['cost_usd']}")

    print(f"\nDone. Cost this run: ${total_cost:.4f}" + (f", {failed} failed" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
