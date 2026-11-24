"""
`scribe serve` -- the chat UI and its API.

Usage:
    scribe serve                      # http://127.0.0.1:8000
    scribe serve --port 8080 --no-rerank

To share it with the party: add logins (scribe users add), then
`tailscale funnel --bg 8000` (see docs/GUIDE.md, "Sharing the site").
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..rag.config import DEFAULT_CONFIG, load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="scribe serve", description="Start the archive's chat UI.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--host", default=None, help="default from rag.toml (127.0.0.1: this computer only)")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--no-warmup", action="store_true",
                        help="load models on the first question instead of at start")
    args = parser.parse_args(argv)

    import uvicorn

    from ..rag.engine import RagEngine
    from ..server.app import create_app

    config = load_config(args.config)
    engine = RagEngine(config, use_reranker=False if args.no_rerank else None)
    app = create_app(engine, warm_up=not args.no_warmup)
    host = args.host or config.server.host
    port = args.port or config.server.port
    from ..server.auth import load_users
    try:
        users = load_users(config.path(config.server.users))
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(f"Archive UI on http://{host}:{port}  (Ctrl+C to stop)")
    if users:
        admins = sum(u.admin for u in users.values())
        print(f"Logins on: {len(users)} user(s), {admins} admin(s).")
    else:
        print("Logins off: open on this computer, refused through a tunnel. "
              "Before sharing: scribe users add NAME --admin")
    uvicorn.run(app, host=host, port=port, log_level="warning")
