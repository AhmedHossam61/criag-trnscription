"""Output writers: transcript.txt / .srt / .json, dropped.txt, corrections."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

from .filters import refresh_drops
from .language import label_script
from .models import Segment


def fmt_timestamp(seconds: float, comma: bool = True) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    sep = "," if comma else "."
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


CORRECTIONS_FILE = "corrections.json"


def apply_corrections(segments: list[Segment], outdir: Path) -> tuple[list[Segment], int]:
    """
    Overlay llm_correct.py's fixes, keyed "<track_index>:<chunk_id>", on copies
    of the segments. segments.jsonl is never changed, so the raw model output
    stays available and a new correction pass can always start over.
    """
    path = outdir / CORRECTIONS_FILE
    if not path.exists():
        return segments, 0
    fixes = json.loads(path.read_text(encoding="utf-8"))
    out, applied = [], 0
    for seg in segments:
        fix = fixes.get(f"{seg.track_index}:{seg.chunk_id}")
        if fix is None or seg.dropped:
            out.append(seg)
            continue
        applied += 1
        if fix.get("drop"):
            out.append(replace(seg, dropped=True, drop_reason="llm-no-speech"))
        else:
            out.append(replace(seg, text=fix["text"], original_text=seg.text))
    return out, applied


def write_transcript_txt(kept: list[Segment], path: Path, info: dict[str, str]) -> None:
    """Readable transcript with consecutive turns by one speaker merged."""
    with path.open("w", encoding="utf-8") as fh:
        if info:
            for key, value in info.items():
                fh.write(f"# {key}: {value}\n")
            fh.write("\n")
        last_speaker = None
        for seg in kept:
            if seg.speaker != last_speaker:
                fh.write(f"\n[{fmt_timestamp(seg.start_s, comma=False)}] {seg.speaker}:\n")
                last_speaker = seg.speaker
            # Mark non-Arabic turns so you can find the code-switches by eye.
            script = label_script(seg.text)
            tag = f"[{script}] " if script in ("en", "mixed") else ""
            fh.write(f"  {tag}{seg.text}\n")


def write_outputs(segments: list[Segment], outdir: Path, info: dict[str, str]) -> None:
    refresh_drops(segments)
    raw_kept = sorted((s for s in segments if not s.dropped), key=lambda s: (s.start_s, s.speaker))
    segments, corrected = apply_corrections(segments, outdir)
    if corrected:
        print(f"  applied {corrected} correction(s) from {CORRECTIONS_FILE}")
        write_transcript_txt(raw_kept, outdir / "transcript-uncorrected.txt", info)

    kept = sorted((s for s in segments if not s.dropped), key=lambda s: (s.start_s, s.speaker))

    (outdir / "transcript.json").write_text(
        json.dumps(
            {"recording": info, "segments": [asdict(s) for s in kept]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    with (outdir / "transcript.srt").open("w", encoding="utf-8") as fh:
        for i, seg in enumerate(kept, start=1):
            fh.write(f"{i}\n")
            fh.write(f"{fmt_timestamp(seg.start_s)} --> {fmt_timestamp(seg.end_s)}\n")
            fh.write(f"{seg.speaker}: {seg.text}\n\n")

    write_transcript_txt(kept, outdir / "transcript.txt", info)

    dropped = [s for s in segments if s.dropped]
    if dropped:
        with (outdir / "dropped.txt").open("w", encoding="utf-8") as fh:
            for seg in sorted(dropped, key=lambda s: s.start_s):
                fh.write(
                    f"[{fmt_timestamp(seg.start_s, comma=False)}] {seg.speaker} "
                    f"({seg.drop_reason}): {seg.text}\n"
                )


def language_report(segments: list[Segment]) -> str:
    """Summarise the Arabic/English split so you can sanity-check the tagging."""
    kept = [s for s in segments if not s.dropped]
    if not kept:
        return ""
    counts = {"ar": 0, "en": 0, "mixed": 0, "?": 0}
    duration = {"ar": 0.0, "en": 0.0, "mixed": 0.0, "?": 0.0}
    for seg in kept:
        key = label_script(seg.text) or "?"
        counts[key] += 1
        duration[key] += seg.end_s - seg.start_s
    total = sum(duration.values()) or 1.0
    parts = [
        f"{key}: {counts[key]} segs / {duration[key] / 60:.1f} min ({100 * duration[key] / total:.0f}%)"
        for key in ("ar", "en", "mixed", "?")
        if counts[key]
    ]
    return "  " + "   ".join(parts)
