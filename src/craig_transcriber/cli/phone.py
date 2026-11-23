"""
`scribe phone` -- Arabic transcription for ONE mixed recording of the
whole table (a phone on the table, a Zoom/Meet export, ...).

Unlike Craig, a phone gives one track with everyone on it, so this script
has to work out who spoke when (diarization) before transcribing:

  1. diarize   -- pyannote (best; needs a free Hugging Face token) or a
                  built-in fallback (silero VAD + voice embeddings + clustering)
  2. name      -- match each anonymous speaker against voice profiles built
                  from your Craig recordings, so SPEAKER_02 becomes "Arnok"
  3. transcribe -- the same QwenCleo batching, filters and resume as
                  scribe craig, one chunk per speaker turn

Usage:
    # once: learn everyone's voice from Craig recordings you already have
    scribe phone --enroll data/recordings/craig-*.zip --speaker-map config/speaker_map.txt

    # every session
    scribe phone data/recordings/session.m4a -o data/transcripts/phone-session --num-speakers 5

    # rename speakers without re-transcribing
    scribe phone --rewrite-outputs -o data/transcripts/phone-session --speaker-map phone_names.txt

Resume is automatic, as in scribe craig. The diarization is cached in
the output folder, so a re-run goes straight to transcription.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..paths import TRANSCRIPTS_DIR
from ..transcription.audio import (
    chunk_peak_db, close_pcm, decode_to_pcm, open_pcm, read_chunk_audio, write_wav,
)
from ..transcription.backends import BACKENDS
from ..transcription.environment import require_ffmpeg
from ..transcription.models import Track
from ..transcription.outputs import language_report
from ..transcription.phone import (
    DEFAULT_VOICES, EXTRA_CLUSTERS, PHONE_MIN_PEAK_DBFS, PYANNOTE_MODEL, Embedder, Turn,
    check_source, diarize_basic, diarize_pyannote, display_names, enroll, load_voices,
    match_speakers, merge_same_voice, print_speakers, speaker_centroids, turns_to_chunks,
    write_named_outputs, write_samples,
)
from ..transcription.pipeline import load_done, make_batches, run_batches
from ..transcription.settings import MIN_GAP_S, SR
from ..transcription.speakers import load_speaker_map


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="scribe phone",
        description="Transcribe one mixed recording of several speakers (Arabic).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1],
    )
    parser.add_argument("inputs", type=Path, nargs="*",
                        help="the recording (any format ffmpeg reads); with --enroll, Craig zips")
    parser.add_argument("-o", "--outdir", type=Path, default=None,
                        help="output folder (default: data/transcripts/phone-<file name>)")
    parser.add_argument("--enroll", action="store_true",
                        help="build voice profiles from Craig recordings instead of transcribing")
    parser.add_argument("--enroll-minutes", type=float, default=5.0, metavar="MIN",
                        help="voice to sample per speaker per Craig recording (default 5)")
    parser.add_argument("--voices", type=Path, default=DEFAULT_VOICES, metavar="FILE",
                        help=f"voice profile file (default {DEFAULT_VOICES})")
    parser.add_argument("--speaker-map", type=Path, default=None, metavar="FILE",
                        help=("with --enroll: Discord username = name, as in scribe craig. "
                              "Otherwise: SPEAKER_00 = name, to fix or set names by hand "
                              "(SKIP hides a speaker)."))
    parser.add_argument("--num-speakers", type=int, default=None, metavar="N",
                        help="how many people talk in the recording -- improves diarization a lot")
    parser.add_argument("--min-speakers", type=int, default=None, metavar="N")
    parser.add_argument("--max-speakers", type=int, default=None, metavar="N")
    parser.add_argument("--diarizer", default="auto", choices=["auto", "pyannote", "basic"],
                        help=("pyannote: best, needs a Hugging Face token. basic: built-in, no "
                              "token, needs --num-speakers or voice profiles. auto (default): "
                              "pyannote if it loads, else basic."))
    parser.add_argument("--hf-token", default=None,
                        help="Hugging Face token (default: $HF_TOKEN, or a saved `hf auth login`)")
    parser.add_argument("--backend", default="qwencleo", choices=sorted(BACKENDS))
    parser.add_argument("--language", default="auto", choices=["ar", "en", "auto"])
    parser.add_argument("--min-gap", type=float, default=MIN_GAP_S, metavar="SECONDS",
                        help=f"join one speaker's turns across pauses up to this long (default {MIN_GAP_S:g})")
    parser.add_argument("--min-peak-db", type=float, default=PHONE_MIN_PEAK_DBFS, metavar="DBFS",
                        help=f"skip chunks whose loudest sample is below this (default {PHONE_MIN_PEAK_DBFS:g})")
    parser.add_argument("--beams", type=int, default=1, metavar="N",
                        help="beam search width, as in scribe craig (default 1)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--batch-seconds", type=float, default=None, metavar="SECONDS")
    parser.add_argument("--limit", type=float, default=None, metavar="SECONDS",
                        help="only process the first N seconds (for testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="diarize and name speakers, but don't transcribe")
    parser.add_argument("--scratch-dir", type=Path, default=None, metavar="DIR")
    parser.add_argument("--rewrite-outputs", action="store_true",
                        help="re-apply names (--speaker-map) and filters to an existing -o folder; no GPU")
    args = parser.parse_args(argv)

    user_map = None
    if args.speaker_map and not args.enroll:
        user_map = load_speaker_map(args.speaker_map)

    if args.enroll:
        if not args.inputs:
            parser.error("--enroll needs one or more Craig recordings")
        require_ffmpeg()
        speaker_map = load_speaker_map(args.speaker_map) if args.speaker_map else None
        enroll(args.inputs, args.voices, speaker_map, args.enroll_minutes, args.scratch_dir)
        return

    if args.rewrite_outputs:
        if args.outdir is None:
            parser.error("--rewrite-outputs needs -o/--outdir")
        jsonl = args.outdir / "segments.jsonl"
        if not jsonl.exists():
            sys.exit(f"error: {jsonl} not found; nothing to rewrite")
        speakers = json.loads((args.outdir / "speakers.json").read_text(encoding="utf-8"))
        info = json.loads((args.outdir / "recording.json").read_text(encoding="utf-8"))
        write_named_outputs(list(load_done(jsonl).values()), args.outdir, info, speakers, user_map)
        names = display_names(speakers, user_map)
        print("\n".join(f"  {lab} -> {names[lab]}" for lab in sorted(names)))
        print(f"Wrote transcript.txt / .srt / .json under {args.outdir}")
        return

    if len(args.inputs) != 1:
        parser.error("give exactly one recording to transcribe")
    src = args.inputs[0]
    if not src.exists():
        sys.exit(f"error: {src} not found")
    require_ffmpeg()
    if args.outdir is None:
        args.outdir = TRANSCRIPTS_DIR / f"phone-{src.stem}"
        print(f"Output folder: {args.outdir}")
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.scratch_dir:
        args.scratch_dir.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="phone_", dir=args.scratch_dir))
    pcm = None

    try:
        info = {
            "Source": src.name,
            "Size bytes": str(src.stat().st_size),
            "Min gap s": f"{args.min_gap:g}",
        }
        if args.limit is not None:
            info["Limit s"] = f"{args.limit:g}"
        check_source(args.outdir, info)

        print(f"Decoding {src.name}...")
        pcm_path = decode_to_pcm(src, scratch / "mix.wav")
        if args.limit is not None:
            trimmed = open_pcm(pcm_path)
            write_wav(np.asarray(trimmed[:int(args.limit * SR)], dtype=np.float32) / 32768.0,
                      scratch / "mix_limit.wav")
            close_pcm(trimmed)
            pcm_path = scratch / "mix_limit.wav"
        pcm = open_pcm(pcm_path)
        print(f"  {len(pcm) / SR / 60:.1f} min of audio")

        voices = load_voices(args.voices)
        if voices:
            print(f"Voice profiles: {', '.join(sorted(voices))} (from {args.voices})")
        else:
            print(f"No voice profiles at {args.voices}: speakers will be named Speaker 1, 2, ... "
                  f"(run --enroll first to name them automatically)")

        # --- 1. diarization (cached) ---------------------------------------
        diar_path = args.outdir / "diarization.json"
        embedder = None
        if diar_path.exists():
            saved = json.loads(diar_path.read_text(encoding="utf-8"))
            turns = [Turn(**t) for t in saved["turns"]]
            info["Diarizer"] = saved["diarizer"]
            print(f"\nUsing saved diarization ({saved['diarizer']}, {len(turns)} turns) "
                  f"from {diar_path}; delete it and segments.jsonl to redo.")
        else:
            print("\nDiarizing (who spoke when)...")
            t0 = time.time()
            turns, used = None, args.diarizer
            if args.diarizer in ("auto", "pyannote"):
                token = args.hf_token or os.environ.get("HF_TOKEN")
                try:
                    turns = diarize_pyannote(pcm, token, args.num_speakers,
                                             args.min_speakers, args.max_speakers)
                    used = "pyannote"
                except Exception as exc:  # noqa: BLE001 -- gated model, no token, ...
                    msg = (
                        f"  pyannote unavailable ({type(exc).__name__}: {str(exc).splitlines()[0][:160]})\n"
                        f"  To enable it: create a free token at https://hf.co/settings/tokens, accept\n"
                        f"  the terms at https://hf.co/{PYANNOTE_MODEL}, then pass --hf-token or set HF_TOKEN."
                    )
                    if args.diarizer == "pyannote":
                        sys.exit(msg)
                    print(msg + "\n  Falling back to the built-in diarizer.")
            if turns is None:
                used = "basic"
                k = args.num_speakers or len(voices)
                if not k:
                    sys.exit("error: the built-in diarizer needs --num-speakers (or voice profiles)")
                if voices:
                    k += EXTRA_CLUSTERS  # merged back by the profile match below
                embedder = Embedder()
                turns = diarize_basic(pcm, k, embedder)
            info["Diarizer"] = used
            n_labels = len({t.label for t in turns})
            print(f"  {len(turns)} turns, {n_labels} speakers ({used}, {time.time() - t0:.0f}s)")
            if voices:
                # Groups that match the same voice are one person: merge them
                # now so their turns join into whole sentences. Saved with the
                # diarization, so chunk ids stay fixed for resume even if the
                # profiles change later.
                embedder = embedder or Embedder()
                matched = match_speakers(speaker_centroids(pcm, turns, embedder), voices)
                turns = merge_same_voice(turns, matched)
                if len({t.label for t in turns}) < n_labels:
                    print(f"  merged into {len({t.label for t in turns})} speakers by voice match")
            diar_path.write_text(json.dumps(
                {"diarizer": used, "turns": [asdict(t) for t in turns]}, indent=1), encoding="utf-8")

        # --- 2. naming -----------------------------------------------------
        speakers_path = args.outdir / "speakers.json"
        labels = sorted({t.label for t in turns})
        if voices:
            embedder = embedder or Embedder()
            speakers = match_speakers(speaker_centroids(pcm, turns, embedder), voices)
        elif speakers_path.exists():
            speakers = json.loads(speakers_path.read_text(encoding="utf-8"))
        else:
            speakers = {}
        for lab in labels:
            speakers.setdefault(lab, {"name": None, "similarity": None, "best": None})
        if embedder:
            embedder.unload()
        speakers_path.write_text(json.dumps(speakers, ensure_ascii=False, indent=2), encoding="utf-8")
        names = display_names(speakers, user_map)
        print_speakers(speakers, turns, names)
        write_samples(pcm, turns, args.outdir)
        print(f"  listen to {args.outdir / 'speaker_samples'} to check; fix names with --speaker-map")

        # --- 3. transcription ----------------------------------------------
        mix = Track(index=0, speaker="mix", path=pcm_path)
        chunks = turns_to_chunks(pcm, turns, args.min_gap)
        loud = [c for c in chunks if chunk_peak_db(pcm, mix, c) >= args.min_peak_db]
        quiet = len(chunks) - len(loud)
        print(f"\n{len(loud)} chunks / {sum(c.duration_s for c in loud) / 60:.1f} min of speech"
              + (f" ({quiet} too quiet, skipped)" if quiet else ""))
        if args.dry_run:
            print("Dry run -- not transcribing.")
            return

        jsonl = args.outdir / "segments.jsonl"
        done = load_done(jsonl)
        if done:
            print(f"Resuming: {len(done)} chunks already transcribed.")
        segments = list(done.values())
        pending = [c for c in loud if c.key not in done]
        failed = 0
        if pending:
            backend = BACKENDS[args.backend](language=args.language)
            backend.beams = args.beams
            batches = make_batches(pending, args.batch_size or max(1, backend.default_batch // args.beams),
                                   args.batch_seconds or backend.default_batch_seconds / args.beams)
            print(f"Loading backend '{backend.name}'; {len(pending)} chunks in {len(batches)} batches.")
            backend.load()
            failed = run_batches(backend, batches,
                                 lambda batch: [read_chunk_audio(pcm, mix, c) for c in batch],
                                 jsonl, segments)
            backend.unload()

        write_named_outputs(segments, args.outdir, info, speakers, user_map)
        kept = sum(1 for s in segments if not s.dropped)
        print(f"\nDone. {kept} segments kept, {len(segments) - kept} filtered.")
        if failed:
            print(f"  {failed} chunk(s) failed -- re-run the same command to retry them.")
        report = language_report(segments)
        if report:
            print(report)
        print(f"Wrote transcript.txt / .srt / .json to {args.outdir}/")

    finally:
        close_pcm(pcm)
        shutil.rmtree(scratch, ignore_errors=True)
