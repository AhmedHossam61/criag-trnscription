"""
`scribe users` -- who can sign in to the lore site.

Usage:
    scribe users add Ahmed --admin     # asks for a password; admins can upload, delete, re-index
    scribe users add Seif              # a member: reads and asks
    scribe users passwd Seif           # a new password (signs Seif out everywhere)
    scribe users remove Seif
    scribe users list

With nobody in config/users.toml the site is open on this computer and refuses
everything that comes through a tunnel. A running `scribe serve` picks up
changes at once.
"""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

from ..rag.config import DEFAULT_CONFIG, load_config
from ..server.auth import User, hash_password, load_users, save_users

MIN_PASSWORD = 8


def ask_password(name: str) -> str:
    while True:
        first = getpass.getpass(f"Password for {name}: ")
        if len(first) < MIN_PASSWORD:
            print(f"  at least {MIN_PASSWORD} characters, please")
            continue
        if getpass.getpass("Again: ") == first:
            return first
        print("  those didn't match, try again")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="scribe users", description="Manage logins for the lore site.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="action", required=True)
    add = sub.add_parser("add", help="add someone (asks for their password)")
    add.add_argument("name")
    add.add_argument("--admin", action="store_true", help="may upload, delete and re-index")
    sub.add_parser("passwd", help="set a new password").add_argument("name")
    sub.add_parser("remove", help="take someone's login away").add_argument("name")
    sub.add_parser("list", help="who can sign in")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    path = config.path(config.server.users)
    if path is None:
        sys.exit("error: [server] users is empty in rag.toml; set it to config/users.toml")
    try:
        users = load_users(path)
    except ValueError as exc:
        sys.exit(f"error: {exc}")

    if args.action == "list":
        if not users:
            print(f"No logins in {path}: the site is open on this computer only.")
        for u in users.values():
            print(f"{u.name:<20} {u.role}")
        return

    name = args.name.strip()
    key = name.lower()
    if args.action == "add":
        if key in users:
            sys.exit(f"error: {users[key].name} already has a login (scribe users passwd {name})")
        users[key] = User(name, "admin" if args.admin else "member", hash_password(ask_password(name)))
        if not any(u.admin for u in users.values()):
            print("Note: nobody is an admin yet, so nobody can change the library from the site.\n"
                  "      Add yourself with --admin.")
        done = f"Added {name} ({users[key].role})."
    elif key not in users:
        sys.exit(f"error: no login called {name} (scribe users list)")
    elif args.action == "passwd":
        users[key].password = hash_password(ask_password(users[key].name))
        done = f"New password set for {users[key].name}; their other sign-ins are ended."
    else:
        done = f"Removed {users.pop(key).name}."
        if not users:
            done += " Nobody is left, so the site is open on this computer only again."
    save_users(path, list(users.values()))
    print(done)
