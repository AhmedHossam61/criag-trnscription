"""
Gemini API keys: where they come from, and how many calls each has left today.

The free tier allows a fixed number of calls per key, per model, per day.
Usage is counted in data/.gemini_usage.json under a fingerprint of each key,
so the correction pass and the question answering share one budget and a
spare key takes over when one runs out.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from ..paths import ENV_FILE, GEMINI_USAGE_FILE


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


def api_keys(cli_key: str | None = None) -> list[str]:
    """
    Every Gemini key available, in the order to use them: --api-key, then
    $GEMINI_API_KEY / $GOOGLE_API_KEY, then .env entries whose name starts
    with GEMINI_API (GEMINI_API, GEMINI_API_2, ...). When one runs out of
    quota the next takes over.
    """
    found = [cli_key, os.environ.get("GEMINI_API_KEY"), os.environ.get("GOOGLE_API_KEY")]
    dotenv = read_dotenv(ENV_FILE)
    found += [dotenv[name] for name in sorted(dotenv) if name.upper().startswith("GEMINI_API")]
    keys: list[str] = []
    for key in found:
        if key and key not in keys:
            keys.append(key)
    return keys


def require_keys(cli_key: str | None = None) -> list[str]:
    keys = api_keys(cli_key)
    if not keys:
        sys.exit("error: no API key. Create one at https://aistudio.google.com/apikey and put it\n"
                 f"  in {ENV_FILE} as GEMINI_API=... (add GEMINI_API_2=... for a second key),\n"
                 "  or set $env:GEMINI_API_KEY, or pass --api-key.")
    return keys


class KeyPool:
    """Hands out the next key that still has calls left today for one model."""

    def __init__(self, keys: list[str], daily_limit: int, model: str,
                 make_client: Callable[[str], object] | None = None,
                 usage_file: Path = GEMINI_USAGE_FILE):
        self.keys = keys
        self.daily_limit = daily_limit
        self.model = model
        self.usage_file = usage_file
        self.exhausted: set[str] = set()  # hit a quota error this run
        self.clients: dict[str, object] = {}
        if make_client is None:
            from .gemini import make_client
        self.make_client = make_client

    def _slot(self, key: str) -> str:
        return f"{fingerprint(key)}/{self.model}"

    def _usage(self) -> dict[str, int]:
        try:
            usage = json.loads(self.usage_file.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            usage = {}
        day = usage.get(quota_day(), {})
        return day if isinstance(day, dict) else {}  # older format counted all keys together

    def left_today(self) -> int:
        usage = self._usage()
        return sum(max(0, self.daily_limit - usage.get(self._slot(k), 0))
                   for k in self.keys if k not in self.exhausted)

    def current(self) -> tuple[str, object] | None:
        usage = self._usage()
        for i, key in enumerate(self.keys):
            if key in self.exhausted or usage.get(self._slot(key), 0) >= self.daily_limit:
                continue
            if key not in self.clients:
                self.clients[key] = self.make_client(key)
            return f"key {i + 1}", self.clients[key]
        return None

    def count_call(self, label: str) -> None:
        slot = self._slot(self.keys[int(label.split()[1]) - 1])
        usage = self._usage()
        usage[slot] = usage.get(slot, 0) + 1
        self.usage_file.parent.mkdir(parents=True, exist_ok=True)
        # Older days are dropped on every write.
        self.usage_file.write_text(json.dumps({quota_day(): usage}), encoding="utf-8")

    def mark_exhausted(self, label: str) -> None:
        self.exhausted.add(self.keys[int(label.split()[1]) - 1])
