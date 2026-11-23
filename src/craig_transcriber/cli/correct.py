"""
`scribe correct` -- fix misheard words in a finished transcript with Gemini.

The speech model gets names, D&D terms and fast slang wrong ("religion check"
-> "الريجيت شيك", "Arnok" -> "ارضك"). This sends Gemini the audio of one window
at a time together with the transcript lines in it and a glossary of your
campaign's words, and asks it to correct only what it hears is wrong -- or to
mark lines where nobody actually spoke (laughs, breathing).

Corrections go to <outdir>/corrections.json. segments.jsonl is never touched;
transcript.txt shows the corrected text and transcript-uncorrected.txt the
original. Built for the free tier's ~20 calls/day: one call per window
(30 min by default, so 12 calls for a 6-hour session), progress is saved after
every call, and a re-run continues with the windows that are left.

API keys are read from .env in the project root (GEMINI_API=..., and
GEMINI_API_2=... etc. as spares that take over when one runs out of quota),
or from $GEMINI_API_KEY / --api-key.

Usage:
    scribe correct -o data/transcripts/v3owGZBtlLQA --audio data/recordings/craig-v3ow....sesx.zip
    scribe correct -o data/transcripts/phone-session1 --audio session.m4a
    scribe correct -o data/transcripts/v3owGZBtlLQA --dry-run       # plan only, no calls
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

from ..correction.corrector import (
    PROGRESS_FILE, AudioSource, ask_gemini, build_prompt, load_json, rewrite, save_json,
    speaker_names,
)
from ..correction.prompts import SYSTEM_AUDIO, SYSTEM_TEXT
from ..llm.gemini import DEFAULT_MODEL, is_quota_error, make_client
from ..llm.keys import KeyPool, api_keys, require_keys
from ..paths import CONFIG_DIR
from ..transcription.environment import require_ffmpeg
from ..transcription.filters import refresh_drops
from ..transcription.outputs import CORRECTIONS_FILE, fmt_timestamp
from ..transcription.pipeline import load_done
from ..transcription.speakers import default_speaker_map, load_speaker_map


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="scribe correct",
        description="Correct a finished transcript with Gemini (listens to the audio).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1],
    )
    parser.add_argument("-o", "--outdir", type=Path, required=False,
                        help="output folder of craig_transcribe.py or phone_transcribe.py")
    parser.add_argument("--audio", type=Path, default=None,
                        help=("the recording: the Craig zip, or the phone audio file. "
                              "Without it Gemini only sees text and fixes far less."))
    parser.add_argument("--speaker-map", type=Path, default=default_speaker_map(), metavar="FILE",
                        help=("Craig only: tracks mapped to SKIP are left out of the audio "
                              "(default config/speaker_map.txt)"))
    parser.add_argument("--glossary", type=Path, default=CONFIG_DIR / "glossary.txt", metavar="FILE",
                        help="names and terms of your campaign (default config/glossary.txt)")
    parser.add_argument("--window-min", type=float, default=None, metavar="MIN",
                        help="minutes of session per call (default 30 with audio, 60 without)")
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL),
                        help=f"Gemini model (default {DEFAULT_MODEL}, or $GEMINI_MODEL)")
    parser.add_argument("--api-key", default=None,
                        help="default: $GEMINI_API_KEY, then GEMINI_API, GEMINI_API_2, ... in .env")
    parser.add_argument("--daily-limit", type=int, default=20, metavar="N",
                        help="calls per day per key, across all folders (default 20)")
    parser.add_argument("--max-calls", type=int, default=None, metavar="N",
                        help="stop after N calls in this run, e.g. 1 to try it out")
    parser.add_argument("--redo", action="store_true",
                        help="discard earlier corrections and start over")
    parser.add_argument("--undo", action="store_true",
                        help="remove all corrections and restore the uncorrected transcript")
    parser.add_argument("--dry-run", action="store_true",
                        help="show the windows and calls needed; no calls")
    parser.add_argument("--list-models", action="store_true",
                        help="list the Gemini models your key can use, then exit")
    args = parser.parse_args(argv)

    if args.list_models:
        client = make_client(require_keys(args.api_key)[0])
        for m in client.models.list():
            if "generateContent" in (getattr(m, "supported_actions", None) or []):
                print(m.name.removeprefix("models/"))
        return
    if args.outdir is None:
        parser.error("-o/--outdir is required")

    outdir = args.outdir
    jsonl = outdir / "segments.jsonl"
    if not jsonl.exists():
        sys.exit(f"error: {jsonl} not found -- transcribe first")
    info = load_json(outdir / "recording.json", {})
    corrections_path = outdir / CORRECTIONS_FILE
    progress_path = outdir / PROGRESS_FILE

    segments = list(load_done(jsonl).values())
    if args.undo:
        corrections_path.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)
        (outdir / "transcript-uncorrected.txt").unlink(missing_ok=True)
        rewrite(outdir, segments, info)
        print("Corrections removed; transcript.txt is the uncorrected version again.")
        return

    refresh_drops(segments)
    names = speaker_names(outdir)
    kept = sorted((s for s in segments if not s.dropped), key=lambda s: s.start_s)
    if not kept:
        sys.exit("error: no transcribed lines to correct")

    use_audio = args.audio is not None
    window_s = (args.window_min or (30.0 if use_audio else 60.0)) * 60
    mode = "audio" if use_audio else "text"
    glossary = args.glossary.read_text(encoding="utf-8").strip() if args.glossary.exists() else ""

    progress = load_json(progress_path, {})
    if args.redo:
        progress = {}
        corrections_path.unlink(missing_ok=True)
    elif progress and (progress.get("window_s") != window_s or progress.get("mode") != mode):
        sys.exit(f"error: {outdir} was corrected with a different window or mode "
                 f"({progress.get('mode')}, {progress.get('window_s', 0) / 60:g} min). "
                 f"Use the same settings to continue, or --redo to start over.")
    progress.update({"window_s": window_s, "mode": mode, "model": args.model})
    done = set(progress.get("done", []))
    corrections = load_json(corrections_path, {})

    n_windows = int(kept[-1].start_s // window_s) + 1
    windows = []
    for w in range(n_windows):
        lines = [s for s in kept if w * window_s <= s.start_s < (w + 1) * window_s]
        if lines:
            windows.append((w, lines))
    pending = [(w, lines) for w, lines in windows if w not in done]

    pool = KeyPool(api_keys(args.api_key), args.daily_limit, args.model, make_client)
    print(f"{len(kept)} lines in {len(windows)} window(s) of {window_s / 60:g} min "
          f"({mode}, model {args.model}); {len(pending)} still to do, "
          f"{len(pool.keys)} API key(s) with {pool.left_today()} call(s) left today.")
    if not glossary:
        print(f"  note: no glossary at {args.glossary} -- names and game terms will be guessed")
    if args.dry_run or not pending:
        if not pending:
            rewrite(outdir, segments, info)
            print("All windows done.")
        return
    require_keys(args.api_key)
    if pool.current() is None:
        sys.exit("Daily call limit reached on every key; run the same command again tomorrow.")

    if use_audio:
        require_ffmpeg()
    speaker_map = load_speaker_map(args.speaker_map) if args.speaker_map else None
    scratch = Path(tempfile.mkdtemp(prefix="llm_"))
    calls = 0
    out_of_calls = False
    try:
        audio_src = AudioSource(args.audio, scratch, speaker_map, info) if use_audio else None
        for w, lines in pending:
            if out_of_calls or (args.max_calls and calls >= args.max_calls):
                print("Stopping here (call limit); run the same command again to continue.")
                break
            start = w * window_s
            end = start + window_s
            audio = audio_src.cut(start, end, w) if audio_src else None
            ids = {}
            text_lines = []
            for i, seg in enumerate(lines, start=1):
                ids[i] = seg
                rel_a, rel_b = seg.start_s - start, seg.end_s - start
                speaker = names.get(seg.speaker, seg.speaker)
                text_lines.append(f"{i} | {fmt_timestamp(rel_a, comma=False)[:-4]}-"
                                  f"{fmt_timestamp(rel_b, comma=False)[:-4]} | {speaker} | {seg.text}")
            speakers = sorted({names.get(s.speaker, s.speaker) for s in lines})
            prompt = build_prompt(glossary, speakers, text_lines, use_audio)

            label = f"{fmt_timestamp(start, comma=False)[:-4]}-{fmt_timestamp(end, comma=False)[:-4]}"
            fixes = None
            while fixes is None:
                picked = pool.current()
                if picked is None:
                    print("no key has calls left.")
                    out_of_calls = True
                    break
                key_label, client = picked
                print(f"  window {label}: {len(lines)} lines ({key_label}) ...", end=" ", flush=True)
                pool.count_call(key_label)
                calls += 1
                try:
                    fixes = ask_gemini(client, args.model,
                                       SYSTEM_AUDIO if use_audio else SYSTEM_TEXT, prompt, audio)
                except Exception as exc:  # noqa: BLE001 -- keep what's done, report, move on
                    if is_quota_error(exc):
                        print(f"{key_label} is out of quota; trying the next key.")
                        pool.mark_exhausted(key_label)
                        continue
                    print(f"failed.\n    {type(exc).__name__}: {str(exc)[:600]}")
                    code = getattr(exc, "code", None)
                    if isinstance(code, int) and 400 <= code < 500:
                        # Wrong model name, bad key, rejected request: every
                        # window would fail the same way, so don't spend more calls.
                        print("    This error won't go away by retrying; stopping. "
                              "Check --model (see --list-models) and the API key.")
                        out_of_calls = True
                    else:
                        print("    Probably temporary; this window will be retried on re-run.")
                    break
            if fixes is None:
                if out_of_calls:
                    break
                continue

            changed = dropped = 0
            for fix in fixes:
                seg = ids.get(fix.get("id"))
                if seg is None:
                    continue  # an id Gemini made up
                key = f"{seg.track_index}:{seg.chunk_id}"
                if fix.get("drop") and use_audio:
                    corrections[key] = {"drop": True, "was": seg.text}
                    dropped += 1
                elif (fix.get("text") or "").strip() and fix["text"].strip() != seg.text.strip():
                    corrections[key] = {"text": fix["text"].strip(), "was": seg.text}
                    changed += 1
            print(f"{changed} corrected, {dropped} marked as no speech")
            done.add(w)
            progress["done"] = sorted(done)
            save_json(corrections_path, corrections)
            save_json(progress_path, progress)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    rewrite(outdir, segments, info)
    remaining = len([w for w, _ in windows if w not in done])
    if corrections:
        print(f"\n{len(corrections)} correction(s) in {corrections_path}.")
        print("transcript.txt is corrected; transcript-uncorrected.txt is the original.")
    elif done:
        print("\nGemini found nothing to correct.")
    else:
        print("\nNo corrections yet -- no window was completed (see the error above).")
    if remaining:
        print(f"{remaining} window(s) left -- run the same command again (tomorrow if the "
              f"daily limit is reached) to continue.")
