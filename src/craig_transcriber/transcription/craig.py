"""Reading Craig Bot recordings: the zip, its tracks, offsets and info.txt."""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from .models import Track


def extract_archive(src: Path, workdir: Path) -> Path:
    """Return a directory containing the unpacked Craig recording."""
    if not src.exists():
        sys.exit(f"error: no such file or directory: {src}")
    if src.is_dir():
        return src
    if src.suffix.lower() != ".zip":
        sys.exit(f"error: expected a .zip or a directory, got {src}")
    dest = workdir / "unpacked"
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src) as zf:
        # raw.dat is Craig's undecoded packet dump (hundreds of MB) -- unused.
        members = [m for m in zf.namelist() if Path(m).name != "raw.dat"]
        zf.extractall(dest, members=members)
    return dest


def parse_sesx_offsets(root: Path) -> dict[str, float]:
    """
    Read per-clip start offsets from Craig's Adobe Audition session file.

    Craig normally pads every track with leading silence so all tracks share
    t=0, in which case every startPoint is 0. But some export modes trim the
    silence and encode the offset here instead, so we honour it.
    """
    offsets: dict[str, float] = {}
    sesx_files = list(root.glob("*.sesx"))
    if not sesx_files:
        return offsets
    try:
        tree = ElementTree.parse(sesx_files[0])
    except ElementTree.ParseError:
        return offsets

    session = tree.getroot().find("session")
    sample_rate = float(session.get("sampleRate", 48000)) if session is not None else 48000.0

    for clip in tree.getroot().iter("audioClip"):
        name = clip.get("name")
        if not name:
            continue
        try:
            offsets[name] = float(clip.get("startPoint", 0)) / sample_rate
        except (TypeError, ValueError):
            offsets[name] = 0.0
    return offsets


def discover_tracks(root: Path) -> list[Track]:
    """Find the per-speaker FLAC files and recover speaker names."""
    audio: list[Path] = []
    for pattern in ("*_data/*.flac", "*_data/*.wav", "*.flac", "*.wav"):
        audio = sorted(root.glob(pattern))
        if audio:
            break
    if not audio:
        sys.exit(f"error: no audio tracks found under {root}")

    offsets = parse_sesx_offsets(root)
    tracks: list[Track] = []
    for path in audio:
        # Craig names files "<index>-<discord username>.flac"
        m = re.match(r"^(\d+)-(.+)$", path.stem)
        if m:
            index, speaker = int(m.group(1)), m.group(2)
        else:
            index, speaker = len(tracks) + 1, path.stem
        tracks.append(
            Track(
                index=index,
                speaker=speaker,
                path=path,
                offset_s=offsets.get(path.name, 0.0),
            )
        )
    tracks.sort(key=lambda t: t.index)
    return tracks


def read_recording_info(root: Path) -> dict[str, str]:
    info_path = root / "info.txt"
    if not info_path.exists():
        return {}
    info: dict[str, str] = {}
    for line in info_path.read_text(encoding="utf-8", errors="replace").splitlines():
        # First line is "Recording <id>" with no colon.
        m = re.match(r"^Recording\s+(\S+)\s*$", line)
        if m:
            info["Recording"] = m.group(1)
        elif ":" in line:
            key, _, value = line.partition(":")
            key, value = key.strip(), value.strip()
            if key and value:
                info[key] = value
    return info
