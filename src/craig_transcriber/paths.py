"""
Where the project keeps its files.

Everything is resolved from the project root (the folder holding
pyproject.toml), so commands work from any working directory. Set
$SCRIBE_HOME to point a run at another root.
"""

from __future__ import annotations

import os
from pathlib import Path


def _find_root() -> Path:
    if os.environ.get("SCRIBE_HOME"):
        return Path(os.environ["SCRIBE_HOME"]).resolve()
    # src/craig_transcriber/paths.py -> project root, for an editable install
    here = Path(__file__).resolve().parents[2]
    if (here / "pyproject.toml").exists():
        return here
    return Path.cwd()


ROOT = _find_root()
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
ENV_FILE = ROOT / ".env"

RECORDINGS_DIR = DATA_DIR / "recordings"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
DOCUMENTS_DIR = DATA_DIR / "documents"
INDEX_DIR = DATA_DIR / "index"

# Gemini calls per key per model today, shared by every command (no keys stored).
GEMINI_USAGE_FILE = DATA_DIR / ".gemini_usage.json"
