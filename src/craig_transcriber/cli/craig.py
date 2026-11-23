"""
`scribe craig` -- Arabic transcription for Craig Bot Discord recordings.

Craig records one FLAC per speaker, all aligned to a common t=0, with exact
digital silence in the gaps (Discord sends no packets when a user isn't
talking). This script exploits both facts:

  * per-speaker tracks  -> diarization for free, no pyannote needed
  * exact-zero gaps     -> a free, perfect VAD; only real speech hits the GPU

Designed for an 8 GB VRAM GPU (RTX 3050) and 8-hour recordings.

Usage:
    scribe craig data/recordings/recording.zip            # -> data/transcripts/<recording id>
    scribe craig recording.zip -o data/experiments/cohere --backend cohere --limit 300
    scribe craig --rewrite-outputs -o data/transcripts/<recording id>
    scribe craig --check-env

Resume is automatic: re-running skips segments already in segments.jsonl.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from ..paths import TRANSCRIPTS_DIR
from ..transcription.audio import close_pcm, open_pcm, read_chunk_audio
from ..transcription.backends import BACKENDS
from ..transcription.craig import extract_archive, read_recording_info
from ..transcription.environment import check_env, require_ffmpeg
from ..transcription.models import Chunk, Segment
from ..transcription.outputs import language_report, write_outputs
from ..transcription.pipeline import (
    LEGACY_MIN_GAP_S, check_recording_match, load_done, make_batches, plan, run_batches,
)
from ..transcription.settings import MIN_GAP_S, MIN_PEAK_DBFS, MIN_SPEECH_RATIO, VAD_MAX_CHUNK_S
from ..transcription.speakers import default_speaker_map, is_skipped, load_speaker_map, map_speaker


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="scribe craig",
        description="Transcribe Craig Bot Discord recordings (Arabic).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input", type=Path, nargs="?",
        help="Craig .zip or an already-extracted directory",
    )
    parser.add_argument(
        "-o", "--outdir", type=Path, default=None,
        help="output folder (default: data/transcripts/<recording id>)",
    )
    parser.add_argument(
        "--backend", default="qwencleo", choices=sorted(BACKENDS),
        help="ASR model to use (default: qwencleo)",
    )
    parser.add_argument(
        "--language", default="auto", choices=["ar", "en", "auto"],
        help=(
            "ar: force the Arabic tag -- right when Arabic is the matrix language "
            "and English appears only as embedded words. en: force English. "
            "auto (default): decide per chunk, for recordings with whole English "
            "sentences mixed in."
        ),
    )
    parser.add_argument(
        "--hotwords", type=Path, default=None, metavar="FILE",
        help="text file of domain vocabulary to bias decoding (qwen3asr, faster-whisper)",
    )
    parser.add_argument(
        "--speaker-map", type=Path, default=default_speaker_map(), metavar="FILE",
        help=(
            "text file mapping Discord usernames to display names, one "
            "'username = display name' per line. Blank lines and '#' comments "
            "are ignored. Match is case-insensitive on the Discord username. "
            "Default: config/speaker_map.txt."
        ),
    )
    parser.add_argument(
        "--min-gap", type=float, default=MIN_GAP_S, metavar="SECONDS",
        help=(
            f"a pause must be at least this long to split a speaker's speech "
            f"into separate chunks (default {MIN_GAP_S:g}). Folders made before "
            f"this option existed used {LEGACY_MIN_GAP_S:g}; pass that to resume them."
        ),
    )
    parser.add_argument(
        "--min-peak-db", type=float, default=MIN_PEAK_DBFS, metavar="DBFS",
        help=(
            f"skip chunks whose loudest sample is below this level "
            f"(default {MIN_PEAK_DBFS:g}). These are mic hiss and breathing that "
            f"the model turns into invented sentences. Raise toward -35 for more "
            f"aggressive filtering; -120 disables it."
        ),
    )
    parser.add_argument(
        "--min-speech-ratio", type=float, default=MIN_SPEECH_RATIO, metavar="R",
        help=(
            f"skip chunks up to {VAD_MAX_CHUNK_S:g} s in which the speech detector hears "
            f"talking for less than this share of the clip (default {MIN_SPEECH_RATIO:g}). "
            f"Catches laughs and breaths the model writes sentences for. "
            f"0 disables it; around 0.35 is stricter but may cut real one-word replies."
        ),
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="max chunks per batch (default: backend's, 16 for qwencleo)",
    )
    parser.add_argument(
        "--batch-seconds", type=float, default=None, metavar="SECONDS",
        help=(
            "max padded audio per batch, i.e. chunk count x longest chunk "
            "(default: backend's, 120 for qwencleo). Lower it on out-of-memory "
            "errors, raise it if VRAM is to spare."
        ),
    )
    parser.add_argument(
        "--limit", type=float, default=None, metavar="SECONDS",
        help=(
            "only transcribe up to N seconds into the recording -- use this to "
            "benchmark backends. With --start, the end of the window."
        ),
    )
    parser.add_argument(
        "--start", type=float, default=None, metavar="SECONDS",
        help=(
            "only transcribe from N seconds into the recording, e.g. --start 11700 "
            "--limit 12000 for 03:15:00-03:20:00. For comparing settings on a short "
            "stretch; use a separate -o folder for each setting."
        ),
    )
    parser.add_argument(
        "--beams", type=int, default=1, metavar="N",
        help=(
            "beam search width (qwencleo only). 1 (default) = greedy, fastest. "
            "4 considers several candidate sentences and keeps the most likely, "
            "usually a bit more accurate, roughly 2-3x slower. Batch defaults "
            "shrink automatically to fit VRAM."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="segment and report only; load no model",
    )
    parser.add_argument(
        "--scratch-dir", type=Path, default=None, metavar="DIR",
        help=(
            "where to put decoded PCM (default: system temp). Budget ~950 MB "
            "per speaker per 8 hours -- point this at a roomy drive, e.g. "
            "D:\\craig-scratch, if your system drive is tight."
        ),
    )
    parser.add_argument("--keep-scratch", action="store_true")
    parser.add_argument(
        "--check-env", action="store_true",
        help="verify ffmpeg, CUDA, VRAM and backend imports, then exit",
    )
    parser.add_argument(
        "--rewrite-outputs", action="store_true",
        help=(
            "re-apply the hallucination filter to an existing "
            "out/segments.jsonl and regenerate transcript.txt/.srt/.json. "
            "No zip, no model, no GPU. Use after updating filter patterns."
        ),
    )
    args = parser.parse_args(argv)

    if args.check_env:
        sys.exit(1 if check_env(args.backend) else 0)
    if args.rewrite_outputs:
        if args.outdir is None:
            parser.error("--rewrite-outputs needs -o/--outdir")
        jsonl = args.outdir / "segments.jsonl"
        if not jsonl.exists():
            sys.exit(f"error: {jsonl} not found; nothing to rewrite")
        done = load_done(jsonl)
        print(f"Loaded {len(done)} segments from {jsonl}")
        if args.speaker_map:
            # Rename segments transcribed before the map had an entry for them.
            speaker_map = load_speaker_map(args.speaker_map)
            for key, seg in list(done.items()):
                renamed = map_speaker(seg.speaker, speaker_map)
                if is_skipped(renamed):
                    del done[key]  # hidden from the transcript; segments.jsonl keeps it
                elif renamed:
                    seg.speaker = renamed
        marker = args.outdir / "recording.json"
        info = json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else {}
        write_outputs(list(done.values()), args.outdir, info)
        print(f"Wrote transcript.txt / .srt / .json under {args.outdir}")
        return
    if args.input is None:
        parser.error("the following arguments are required: input")

    require_ffmpeg()
    if args.scratch_dir:
        args.scratch_dir.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="craig_", dir=args.scratch_dir))

    try:
        root = extract_archive(args.input, scratch)
        info = read_recording_info(root)
        if info:
            print("Recording:")
            for key, value in info.items():
                print(f"  {key}: {value}")
        if args.outdir is None:
            args.outdir = TRANSCRIPTS_DIR / (info.get("Recording") or args.input.stem)
        print(f"Output folder: {args.outdir}")
        args.outdir.mkdir(parents=True, exist_ok=True)
        if not args.dry_run:
            check_recording_match(args.outdir, info, args.min_gap)

        speaker_map = load_speaker_map(args.speaker_map) if args.speaker_map else None
        if speaker_map:
            print(f"\nSpeaker map: {len(speaker_map)} rename(s) loaded from {args.speaker_map}")

        print("\nSegmenting tracks (zero-run VAD):")
        t0 = time.time()
        tracks, plans, pcm_paths, skipped = plan(
            root, scratch, args.limit, speaker_map=speaker_map,
            min_peak_db=args.min_peak_db, min_gap_s=args.min_gap, start_s=args.start,
            min_speech_ratio=args.min_speech_ratio,
        )
        total_chunks = sum(len(c) for c in plans.values())
        total_speech = sum(c.duration_s for cs in plans.values() for c in cs)
        print(
            f"\n  {total_chunks} chunks / {total_speech / 60:.1f} min of speech "
            f"(segmented in {time.time() - t0:.1f}s)"
        )

        if args.dry_run:
            print("\nDry run -- no model loaded.")
            return

        hotwords = None
        if args.hotwords:
            hotwords = args.hotwords.read_text(encoding="utf-8").strip()

        backend = BACKENDS[args.backend](language=args.language, hotwords=hotwords)
        backend.beams = args.beams
        if hotwords and not backend.supports_hotwords:
            print(f"  note: backend '{backend.name}' ignores --hotwords")
        if args.beams > 1 and backend.name != "qwencleo":
            print(f"  note: --beams is only used by qwencleo; '{backend.name}' ignores it")
        # Each beam is another copy of the batch on the GPU.
        batch_size = args.batch_size or max(1, backend.default_batch // args.beams)
        batch_seconds = args.batch_seconds or backend.default_batch_seconds / args.beams

        jsonl = args.outdir / "segments.jsonl"
        done = load_done(jsonl)
        if done:
            print(f"\nResuming: {len(done)} segments already transcribed.")
        # Segments transcribed before the level gate or speech check existed
        # (or under looser thresholds) are hidden from the outputs;
        # segments.jsonl keeps them.
        for key, seg in done.items():
            if key in skipped and not seg.dropped:
                seg.dropped, seg.drop_reason = True, skipped[key]

        segments: list[Segment] = list(done.values())
        pending = [c for cs in plans.values() for c in cs if c.key not in done]
        if pending:
            print(
                f"\nLoading backend '{backend.name}' (batch up to {batch_size} chunks / "
                f"{batch_seconds:g}s padded audio, language={args.language})..."
            )
            backend.load()
        batches = make_batches(pending, batch_size, batch_seconds)
        print(f"{len(pending)} chunks to transcribe in {len(batches)} batches (longest first).")

        track_by_index = {t.index: t for t in tracks}
        pcms = {
            t.index: open_pcm(pcm_paths[t.index])
            for t in tracks if any(c.key not in done for c in plans[t.index])
        }

        def load_batch(batch: list[Chunk]) -> list[np.ndarray]:
            return [
                read_chunk_audio(pcms[c.track_index], track_by_index[c.track_index], c)
                for c in batch
            ]

        try:
            failed = run_batches(backend, batches, load_batch, jsonl, segments)
        finally:
            for pcm in pcms.values():
                close_pcm(pcm)

        backend.unload()
        write_outputs(segments, args.outdir, info)

        kept = sum(1 for s in segments if not s.dropped)
        n_quiet = sum(1 for r in skipped.values() if r == "too-quiet")
        print(
            f"\nDone. {kept} segments kept, {len(segments) - kept} filtered; skipped "
            f"{n_quiet} chunks as too quiet (below {args.min_peak_db:g} dBFS) and "
            f"{len(skipped) - n_quiet} with too little speech."
        )
        if failed:
            print(f"  {failed} chunk(s) failed -- re-run the same command to retry them.")
        report = language_report(segments)
        if report:
            print("Script mix (by output, not by requested tag):")
            print(report)
        print(f"Wrote transcript.txt / .srt / .json to {args.outdir}/")

    finally:
        if args.keep_scratch:
            print(f"Scratch retained at {scratch}")
        else:
            shutil.rmtree(scratch, ignore_errors=True)
            if scratch.exists():
                # Windows refuses to unlink a file that is still mapped.
                print(
                    f"warning: could not fully remove scratch at {scratch} -- "
                    f"delete it manually to reclaim the disk space."
                )
