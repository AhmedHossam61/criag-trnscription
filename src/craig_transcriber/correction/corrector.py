"""
The Gemini correction pass: cut the audio of one window, send it with the
transcript lines in it, and read back only the lines that change.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from ..transcription.craig import discover_tracks, extract_archive, read_recording_info
from ..transcription.models import Segment
from ..transcription.outputs import write_outputs
from ..transcription.speakers import is_skipped, map_speaker


PROGRESS_FILE = "llm_progress.json"
INLINE_LIMIT = 14 * 1024 * 1024  # larger audio goes through the Files API


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
                if is_skipped(name):
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


def speaker_names(outdir: Path) -> dict[str, str]:
    """Phone transcripts store SPEAKER_00 labels; show the names instead."""
    path = outdir / "speakers.json"
    if not path.exists():
        return {}
    from ..transcription.phone import display_names
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
        from ..transcription.phone import write_named_outputs
        write_named_outputs(segments, outdir, info,
                            json.loads(speakers_path.read_text(encoding="utf-8")), None)
    else:
        write_outputs(segments, outdir, info)
