#!/usr/bin/env python3
"""
llm_correct.py -- fix misheard words in a finished transcript with Gemini.

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

API keys are read from .env next to this script (GEMINI_API=..., and
GEMINI_API_2=... etc. as spares that take over when one runs out of quota),
or from $GEMINI_API_KEY / --api-key.

Usage:
    python llm_correct.py -o out-v3owGZBtlLQA --audio "craig-v3ow....sesx.zip" --speaker-map speaker_map.txt
    python llm_correct.py -o out-phone-session1 --audio session.m4a
    python llm_correct.py -o out-v3owGZBtlLQA --dry-run       # plan only, no calls
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from craig_transcribe import (
    CORRECTIONS_FILE, Segment,
    discover_tracks, extract_archive, fmt_timestamp, load_done, load_speaker_map,
    map_speaker, read_recording_info, refresh_drops, require_ffmpeg, write_outputs,
)

DEFAULT_MODEL = "gemini-2.5-flash"
PROGRESS_FILE = "llm_progress.json"
USAGE_FILE = Path(__file__).with_name(".gemini_usage.json")  # shared by all output folders
SKIP_NAMES = {"SKIP", "-", "IGNORE"}
INLINE_LIMIT = 14 * 1024 * 1024  # larger audio goes through the Files API

SYSTEM_AUDIO = """\
You correct an automatic transcript of a Dungeons & Dragons session played in \
Egyptian Arabic with a lot of English mixed in. You get the audio of one \
window of the session and the transcript lines that fall inside it.

Each line is: id | start-end (time inside THIS audio) | speaker | text.
Every player had their own microphone, so the speaker label is always right. \
Only the text can be wrong.

For each line, listen to that speaker at that time and decide:
- The text is right, or close enough that you are unsure: do NOT list it.
- Words are misheard: return {"id": <id>, "text": "<the full corrected line>"}.
- That speaker says nothing there (only breathing, laughter, noise, or other \
people talking) and the text is invented: return {"id": <id>, "drop": true}.

Rules:
- Write what was actually said, not what should have been said. Never \
improve, summarise, translate or complete a sentence.
- Keep Egyptian dialect spelling as spoken (ايه، عايز، مش، كده). Do not \
convert to Modern Standard Arabic.
- English words and phrases go in Latin script ("religion check", "spell \
slots", "long rest"), never in Arabic letters.
- Keep slang and profanity exactly as said; do not censor.
- Use the glossary for the spelling of names and game terms.
- Do not merge or split lines, and do not add speech from outside a line's \
time span.
- If speech overlaps or is unclear, leave the line unchanged.
Return a JSON array with only the lines you change. Return [] if none."""

SYSTEM_TEXT = """\
You correct an automatic transcript of a Dungeons & Dragons session played in \
Egyptian Arabic with a lot of English mixed in. You do NOT have the audio, \
so be conservative.

Each line is: id | start-end | speaker | text.

Only fix words that are clearly mis-recognitions of:
- names and game terms from the glossary ("ارضك" for "Arnok" when he is \
addressed),
- English words or phrases written phonetically in Arabic letters ("ام \
ترينج تو جيس" -> "I'm trying to guess").
Return {"id": <id>, "text": "<the full corrected line>"} for each line you \
change. Leave everything else exactly as it is, including slang and \
profanity. Never rewrite style, translate, or guess at garbled lines.
Return a JSON array with only the lines you change. Return [] if none."""


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------

def quota_day() -> str:
    """Gemini's free-tier daily limits reset at midnight Pacific time."""
    return datetime.now(timezone(timedelta(hours=-8))).strftime("%Y-%m-%d")


def fingerprint(key: str) -> str:
    """Usage is tracked per key without writing the key itself to disk."""
    return hashlib.sha256(key.encode()).hexdigest()[:10]


def read_dotenv(path: Path) -> dict[str, str]:
    """Minimal .env reader: NAME=value lines, optional quotes, '#' comments."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        values[name.strip().removeprefix("export ").strip()] = value.strip().strip("'\"")
    return values


def api_keys(cli_key: str | None) -> list[str]:
    """
    Every Gemini key available, in the order to use them: --api-key, then
    $GEMINI_API_KEY / $GOOGLE_API_KEY, then .env entries whose name starts
    with GEMINI_API (GEMINI_API, GEMINI_API_2, ...). When one runs out of
    quota the next takes over.
    """
    found = [cli_key, os.environ.get("GEMINI_API_KEY"), os.environ.get("GOOGLE_API_KEY")]
    dotenv = read_dotenv(Path(__file__).with_name(".env"))
    found += [dotenv[name] for name in sorted(dotenv) if name.upper().startswith("GEMINI_API")]
    keys: list[str] = []
    for key in found:
        if key and key not in keys:
            keys.append(key)
    return keys


class KeyPool:
    """Hands out the next key that still has calls left today."""

    def __init__(self, keys: list[str], daily_limit: int):
        self.keys = keys
        self.daily_limit = daily_limit
        self.exhausted: set[str] = set()  # hit a quota error this run
        self.clients: dict[str, object] = {}

    def _usage(self) -> dict[str, int]:
        usage = json.loads(USAGE_FILE.read_text(encoding="utf-8")) if USAGE_FILE.exists() else {}
        day = usage.get(quota_day(), {})
        return day if isinstance(day, dict) else {}  # older format counted all keys together

    def left_today(self) -> int:
        usage = self._usage()
        return sum(max(0, self.daily_limit - usage.get(fingerprint(k), 0))
                   for k in self.keys if k not in self.exhausted)

    def current(self) -> tuple[str, object] | None:
        usage = self._usage()
        for i, key in enumerate(self.keys):
            if key in self.exhausted or usage.get(fingerprint(key), 0) >= self.daily_limit:
                continue
            if key not in self.clients:
                from google import genai
                self.clients[key] = genai.Client(api_key=key)
            return f"key {i + 1}", self.clients[key]
        return None

    def count_call(self, label: str) -> None:
        key = self.keys[int(label.split()[1]) - 1]
        usage = self._usage()
        usage[fingerprint(key)] = usage.get(fingerprint(key), 0) + 1
        USAGE_FILE.write_text(json.dumps({quota_day(): usage}), encoding="utf-8")  # older days dropped

    def mark_exhausted(self, label: str) -> None:
        self.exhausted.add(self.keys[int(label.split()[1]) - 1])


def load_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def save_json(path: Path, data) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)  # never leave a half-written file if killed mid-save


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

class AudioSource:
    """Cuts one window of the session to a small mono MP3 for Gemini."""

    def __init__(self, src: Path, scratch: Path, speaker_map: dict[str, str] | None,
                 outdir_info: dict[str, str]):
        self.scratch = scratch
        self.inputs: list[tuple[Path, float]] = []  # (file, offset on the session timeline)
        if src.suffix.lower() == ".zip" or src.is_dir():
            root = extract_archive(src, scratch)
            rec = read_recording_info(root).get("Recording")
            if rec and outdir_info.get("Recording") and rec != outdir_info["Recording"]:
                sys.exit(f"error: {src.name} is recording {rec}, but the output folder "
                         f"holds {outdir_info['Recording']}")
            for track in discover_tracks(root):
                name = map_speaker(track.speaker, speaker_map) if speaker_map else None
                if name and name.upper() in SKIP_NAMES:
                    continue
                self.inputs.append((track.path, track.offset_s))
            print(f"Audio: {len(self.inputs)} Craig track(s) mixed from {src.name}")
        else:
            source = outdir_info.get("Source")
            if source and source != src.name:
                sys.exit(f"error: the output folder was made from {source}, not {src.name}")
            self.inputs.append((src, 0.0))
            print(f"Audio: {src.name}")

    def cut(self, start_s: float, end_s: float, index: int) -> Path:
        dest = self.scratch / f"window{index:03d}.mp3"
        length = end_s - start_s
        cmd = ["ffmpeg", "-v", "error", "-y"]
        labels = []
        for i, (path, offset) in enumerate(self.inputs):
            if offset >= end_s:
                continue  # track starts after this window
            seek = max(0.0, start_s - offset)
            delay_ms = int(max(0.0, offset - start_s) * 1000)
            cmd += ["-ss", f"{seek:.3f}", "-t", f"{length:.3f}", "-i", str(path)]
            labels.append((len(labels), delay_ms))
        chains = [f"[{i}:a]adelay={d}:all=1[a{i}]" for i, d in labels]
        mix = "".join(f"[a{i}]" for i, _ in labels)
        graph = ";".join(chains + [f"{mix}amix=inputs={len(labels)}:normalize=0,alimiter=limit=0.95[out]"])
        cmd += ["-filter_complex", graph, "-map", "[out]",
                "-ac", "1", "-ar", "16000", "-c:a", "libmp3lame", "-b:a", "32k", str(dest)]
        subprocess.run(cmd, check=True)
        return dest


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

def require_keys(cli_key: str | None) -> list[str]:
    keys = api_keys(cli_key)
    if not keys:
        sys.exit("error: no API key. Create one at https://aistudio.google.com/apikey and put it\n"
                 "  in .env next to this script as GEMINI_API=... (add GEMINI_API_2=... for a\n"
                 "  second key), or set $env:GEMINI_API_KEY, or pass --api-key.")
    return keys


def ask_gemini(client, model: str, system: str, prompt: str, audio: Path | None) -> list[dict]:
    from google.genai import types

    contents: list = []
    if audio is not None:
        if audio.stat().st_size <= INLINE_LIMIT:
            contents.append(types.Part.from_bytes(data=audio.read_bytes(), mime_type="audio/mp3"))
        else:
            contents.append(client.files.upload(file=str(audio)))
    contents.append(prompt)

    # Players swear; the default filters would block whole windows over it.
    safety = [
        types.SafetySetting(category=c, threshold=types.HarmBlockThreshold.OFF)
        for c in (types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                  types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                  types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                  types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT)
    ]
    schema = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "id": {"type": "INTEGER"},
                "text": {"type": "STRING"},
                "drop": {"type": "BOOLEAN"},
            },
            "required": ["id"],
        },
    }
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.2,
            safety_settings=safety,
        ),
    )
    if not response.text:
        reason = response.candidates[0].finish_reason if response.candidates else "no candidates"
        raise RuntimeError(f"empty response ({reason})")
    fixes = json.loads(response.text)
    if not isinstance(fixes, list):
        raise RuntimeError("response is not a JSON array")
    return fixes


def is_quota_error(exc: Exception) -> bool:
    return getattr(exc, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(exc)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def speaker_names(outdir: Path) -> dict[str, str]:
    """Phone transcripts store SPEAKER_00 labels; show the names instead."""
    path = outdir / "speakers.json"
    if not path.exists():
        return {}
    from phone_transcribe import display_names
    return display_names(json.loads(path.read_text(encoding="utf-8")), None)


def build_prompt(glossary: str, speakers: list[str], lines: list[str], audio: bool) -> str:
    parts = []
    if glossary:
        parts.append("Glossary (names and terms used in this campaign):\n" + glossary)
    parts.append("Speakers: " + ", ".join(speakers))
    what = "the attached audio" if audio else "this part of the session"
    parts.append(f"Transcript lines for {what}:\n" + "\n".join(lines))
    return "\n\n".join(parts)


def rewrite(outdir: Path, segments: list[Segment], info: dict[str, str]) -> None:
    speakers_path = outdir / "speakers.json"
    if speakers_path.exists():
        from phone_transcribe import write_named_outputs
        write_named_outputs(segments, outdir, info,
                            json.loads(speakers_path.read_text(encoding="utf-8")), None)
    else:
        write_outputs(segments, outdir, info)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Correct a finished transcript with Gemini (listens to the audio).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1],
    )
    parser.add_argument("-o", "--outdir", type=Path, required=False,
                        help="output folder of craig_transcribe.py or phone_transcribe.py")
    parser.add_argument("--audio", type=Path, default=None,
                        help=("the recording: the Craig zip, or the phone audio file. "
                              "Without it Gemini only sees text and fixes far less."))
    parser.add_argument("--speaker-map", type=Path, default=None, metavar="FILE",
                        help="Craig only: tracks mapped to SKIP are left out of the audio")
    parser.add_argument("--glossary", type=Path, default=Path("glossary.txt"), metavar="FILE",
                        help="names and terms of your campaign (default glossary.txt)")
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
    args = parser.parse_args()

    if args.list_models:
        from google import genai
        client = genai.Client(api_key=require_keys(args.api_key)[0])
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
        print(f"Corrections removed; transcript.txt is the uncorrected version again.")
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

    pool = KeyPool(api_keys(args.api_key), args.daily_limit)
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
                    print(f"failed ({type(exc).__name__}: {str(exc)[:200]}); will retry on re-run")
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
    print(f"\n{len(corrections)} correction(s) in {corrections_path}.")
    print(f"transcript.txt is corrected; transcript-uncorrected.txt is the original.")
    if remaining:
        print(f"{remaining} window(s) left -- run the same command again (tomorrow if the "
              f"daily limit is reached) to continue.")


if __name__ == "__main__":
    main()
