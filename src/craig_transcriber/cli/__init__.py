"""
`scribe` -- one command for the whole project.

    scribe craig     transcribe a Craig Bot recording (one track per player)
    scribe phone     transcribe one mixed recording (phone, Zoom), or --enroll voices
    scribe correct   fix misheard lines with Gemini, listening to the audio
    scribe import    add an old transcript.txt as a session
    scribe rag       build the document index and ask questions from the terminal
    scribe serve     start the chat UI at http://127.0.0.1:8000

`scribe <command> --help` shows each command's options.
"""

from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "craig": "craig_transcriber.cli.craig",
    "phone": "craig_transcriber.cli.phone",
    "correct": "craig_transcriber.cli.correct",
    "import": "craig_transcriber.cli.importer",
    "rag": "craig_transcriber.cli.rag",
    "serve": "craig_transcriber.cli.serve",
}


def force_utf8_console() -> None:
    """
    Craig metadata carries Arabic (guild names, channel names, usernames) and
    transcripts obviously do. Windows CMD defaults to cp1252/cp437, where
    printing Arabic raises UnicodeEncodeError and kills the run. Files are
    always written as UTF-8; this fixes stdout/stderr too.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv: list[str] | None = None) -> None:
    force_utf8_console()
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__.strip())
        return
    command, rest = argv[0], argv[1:]
    if command in ("-V", "--version"):
        from .. import __version__
        print(__version__)
        return
    if command not in COMMANDS:
        sys.exit(f"unknown command '{command}'. Commands: {', '.join(COMMANDS)}")
    importlib.import_module(COMMANDS[command]).main(rest)
