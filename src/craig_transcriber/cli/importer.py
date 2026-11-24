"""
`scribe import` -- add an old transcript.txt as a session.

Usage:
    scribe import "old\\transcript.txt"                 # -> data/transcripts/<recording id>/
    scribe import old\\*.txt                            # several at once
    scribe import notes.txt --id session-05 --date 2026-03-14

For transcripts made by an earlier version of this tool when only the text is
left (no segments.jsonl): lines like

    [00:00:06.299] Dungeon Master:
      [mixed] الcult مصدقين إن هما ...

or "[00:00:06] Dungeon Master: text" on one line. Each becomes a session folder
with a transcript.json, so the archive and the character pages treat it like
any other session. Speakers go through config/speaker_map.txt (Discord names
-> characters, SKIP drops a bot). Only each turn's start time is in the text;
the times of the lines inside a turn are estimated from their length.

Text in any other shape belongs in data/documents/sessions instead: it is
still searched and cited, it just doesn't count as a session.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

from ..paths import DOCUMENTS_DIR, TRANSCRIPTS_DIR
from ..transcription.speakers import default_speaker_map, is_skipped, load_speaker_map, map_speaker

HEADER = re.compile(r"^#(?!#)\s*([^:`]+?)\s*:\s*(.*)$")       # "# Start time: ...", not "### name `0:01`"
CRAIG_TITLE = re.compile(r"^#\s*Craig Transcript\s+(\w+)\s*$", re.I)
_TIME = r"(\d+(?::\d{2}){1,2}(?:[.,]\d+)?)"
# The turn layouts seen so far, each giving start, optional end, speaker, text:
TURNS = [
    # [00:00:06.299] Dungeon Master:          (this tool; text on the next lines, or after the colon)
    re.compile(rf"^\[{_TIME}\]\s*(?P<speaker>[^:\[\]]*?)\s*:\s*(?P<rest>.*)$"),
    # [00:00:01.460 - 00:00:03.460] **seifeldien:** text
    re.compile(rf"^\[{_TIME}\s*-\s*{_TIME}\]\s*\*\*(?P<speaker>[^*]+?):?\*\*:?\s*(?P<rest>.*)$"),
    # `00:02` **hazemonsif:** text
    re.compile(rf"^`{_TIME}`\s*\*\*(?P<speaker>[^*]+?):?\*\*:?\s*(?P<rest>.*)$"),
    # ### venom6469 `0:00:03`                 (text on the next lines)
    re.compile(rf"^#{{2,4}}\s*(?P<speaker>.+?)\s*`{_TIME}`\s*$"),
]
SCRIPT_TAG = re.compile(r"^\[(?:en|mixed|ar)\]\s*")
WORDS_PER_SECOND = 2.5


def seconds(stamp: str) -> float:
    """'1:02:03.5', '02:03' -> seconds."""
    parts = stamp.replace(",", ".").split(":")
    total = 0.0
    for part in parts:
        total = total * 60 + float(part)
    return total


def parse_transcript_txt(text: str) -> tuple[dict[str, str], list[dict]]:
    """Header fields and turns ({speaker, start_s, end_s, lines}) of a transcript."""
    info: dict[str, str] = {}
    turns: list[dict] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if not turns and (m := CRAIG_TITLE.match(line)):
            info["Recording"] = m.group(1)
            continue
        if not turns and (m := HEADER.match(line)):
            info[m.group(1)] = m.group(2).strip()
            continue
        m = next((m for pattern in TURNS if (m := pattern.match(line))), None)
        if m:
            stamps = [g for g in m.groups() if g and re.fullmatch(_TIME.strip("()"), g)]
            turns.append({"speaker": m.group("speaker").strip(), "start_s": seconds(stamps[0]),
                          "end_s": seconds(stamps[1]) if len(stamps) > 1 else None, "lines": []})
            rest = m.groupdict().get("rest") or ""
            if rest.strip():
                turns[-1]["lines"].append(SCRIPT_TAG.sub("", rest.strip()))
        elif turns and line != "---":
            turns[-1]["lines"].append(SCRIPT_TAG.sub("", line))
    return info, [t for t in turns if t["lines"]]


def to_segments(turns: list[dict], speaker_map: dict[str, str]) -> list[dict]:
    """One segment per line; times inside a turn estimated from word counts."""
    turns = sorted(turns, key=lambda t: t["start_s"])
    segments = []
    for i, turn in enumerate(turns):
        mapped = map_speaker(turn["speaker"], speaker_map) if turn["speaker"] else None
        if is_skipped(mapped):
            continue
        speaker = mapped or turn["speaker"] or "Unknown"
        lengths = [max(1.0, len(line.split()) / WORDS_PER_SECOND) for line in turn["lines"]]
        next_start = next((t["start_s"] for t in turns[i + 1:] if t["start_s"] > turn["start_s"]), None)
        if turn.get("end_s") and turn["end_s"] > turn["start_s"]:       # the file gives the end: fill it
            scale = (turn["end_s"] - turn["start_s"]) / sum(lengths)
        else:                                                           # estimate, but stop at the next turn
            room = (next_start - turn["start_s"]) if next_start else sum(lengths)
            scale = min(1.0, room / sum(lengths))
        t = turn["start_s"]
        for line, length in zip(turn["lines"], lengths):
            segments.append({"speaker": speaker, "start_s": round(t, 3), "end_s": round(t + length * scale, 3),
                             "text": line, "language": "", "dropped": False, "imported": True})
            t += length * scale
    return segments


def _raw_names(turns: list[dict], speaker_map: dict[str, str]) -> set[str]:
    """Speaker labels the map doesn't know and that aren't already a mapped name."""
    known = {v.lower() for v in speaker_map.values()}
    return {t["speaker"].lower() for t in turns
            if t["speaker"] and map_speaker(t["speaker"], speaker_map) is None and t["speaker"].lower() not in known}


def import_file(path: Path, args, speaker_map: dict[str, str]) -> bool:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    info, turns = parse_transcript_txt(text)
    if not turns:
        print(f"{path}: no '[00:00:00] Speaker:' lines found, so this is notes rather than a transcript.")
        if path.resolve().is_relative_to(DOCUMENTS_DIR.resolve()):
            print("  It's already in data/documents, so nothing to import: `scribe rag ingest` "
                  "(or Re-index in the site) adds it, and answers can cite it.")
        else:
            print("  Put it in data/documents/sessions instead: it will still be searched and cited.")
        return False
    if args.date:
        info["Start time"] = args.date
    folder_id = args.id or info.get("Recording") or re.sub(r"[^\w.-]+", "-", path.stem).strip("-")
    info.setdefault("Recording", folder_id)
    out = args.output_root / folder_id

    if (out / "segments.jsonl").exists():
        print(f"{path}: {out} is a full transcription of the same recording; keeping it "
              f"(it is better than the text). Use --id to import under another name anyway.")
        return False
    if (out / "transcript.json").exists() and not args.force:
        print(f"{path}: {out} already exists; --force replaces it.")
        return False

    segments = to_segments(turns, speaker_map)
    out.mkdir(parents=True, exist_ok=True)
    info |= {"Imported from": path.name, "Imported": time.strftime("%Y-%m-%d"),
             "Note": "imported from text; times inside a turn are estimated"}
    (out / "recording.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "transcript.json").write_text(json.dumps({"recording": info, "segments": segments},
                                                    ensure_ascii=False, indent=1), encoding="utf-8")
    # Keep the original next to what was made from it. One sitting in
    # data/documents moves out of there, or it would be indexed twice.
    original = out / f"original-{path.name}"
    moved = path.resolve().is_relative_to(DOCUMENTS_DIR.resolve())
    # Re-importing the original kept in the folder (--force) leaves it where it is.
    if path.resolve().parent != out.resolve():
        (shutil.move if moved else shutil.copyfile)(path, original)

    speakers: dict[str, int] = {}
    for seg in segments:
        speakers[seg["speaker"]] = speakers.get(seg["speaker"], 0) + 1
    date = (info.get("Start time") or "no date")[:10]
    print(f"{path.name} -> {out}  ({date}, {len(segments)} lines)")
    print("  " + ", ".join(f"{name} {n}" for name, n in sorted(speakers.items(), key=lambda kv: -kv[1])))
    if moved:
        print(f"  moved the original out of data/documents to {original.name} in that folder")
    unknown = sorted(s for s in speakers if speaker_map and s.lower() in _raw_names(turns, speaker_map))
    if unknown:
        print(f"  not in the speaker map (shown as-is): {', '.join(unknown)}. Add them to "
              f"config/speaker_map.txt and import again with --force to rename.")
    return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="scribe import", description="Add old transcript.txt files as sessions.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__.split("Usage:")[1])
    parser.add_argument("files", nargs="+", type=Path, help="transcript .txt files")
    parser.add_argument("--id", help="folder name (default: the '# Recording:' line, else the file name)")
    parser.add_argument("--date", help="session date, YYYY-MM-DD, when the file has no '# Start time:' line")
    parser.add_argument("--speaker-map", type=Path, default=default_speaker_map(),
                        help="Discord username -> name (default config/speaker_map.txt)")
    parser.add_argument("--output-root", type=Path, default=TRANSCRIPTS_DIR, help=argparse.SUPPRESS)
    parser.add_argument("--force", action="store_true", help="replace an earlier import of the same session")
    args = parser.parse_args(argv)
    if args.id and len(args.files) > 1:
        parser.error("--id names one session; import the files one at a time")

    speaker_map = load_speaker_map(args.speaker_map) if args.speaker_map else {}
    done = sum(import_file(p, args, speaker_map) for p in args.files)
    if not done:
        sys.exit(1)
    print(f"\nImported {done} of {len(args.files)}. Run `scribe rag ingest` (or Re-index in the site) to add them.")
