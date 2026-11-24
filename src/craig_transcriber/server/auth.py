"""
Logins for the lore site.

With no user in config/users.toml the site is open, as it always was on
127.0.0.1. With one or more, every page and API call needs a login, and only
admins can upload, delete or re-index. `scribe users` manages the file.

Passwords are kept as scrypt hashes, never as text. A login is a signed cookie
that lasts `session_days`; it carries a piece of the password hash, so changing
someone's password signs them out everywhere.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROLES = ("admin", "member")
SECRET_ENV = "SCRIBE_SESSION_SECRET"
_SCRYPT = {"n": 2 ** 14, "r": 8, "p": 1}
MAX_FAILURES = 8           # wrong passwords per name ...
FAILURE_WINDOW_S = 15 * 60  # ... in this long, then that name is locked until the window passes


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, **_SCRYPT)
    return f"scrypt${salt.hex()}${digest.hex()}"


def check_password(password: str, stored: str) -> bool:
    try:
        scheme, salt, digest = stored.split("$")
        if scheme != "scrypt":
            return False
        got = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), **_SCRYPT)
    except ValueError:
        return False
    return hmac.compare_digest(got.hex(), digest)


@dataclass
class User:
    name: str
    role: str = "member"
    password: str = ""     # the scrypt hash

    @property
    def admin(self) -> bool:
        return self.role == "admin"


def load_users(path: Path | None) -> dict[str, User]:
    """Users by lower-cased name. Raises ValueError on a mistake in the file."""
    if path is None or not path.exists():
        return {}
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{path.name}: {exc}") from exc
    users: dict[str, User] = {}
    for i, entry in enumerate(data.get("user", []), 1):
        unknown = set(entry) - {"name", "role", "password"}
        if unknown:
            raise ValueError(f"{path.name}: user {i} has unknown setting(s) {', '.join(sorted(unknown))}")
        name = str(entry.get("name", "")).strip()
        role = entry.get("role", "member")
        if not name:
            raise ValueError(f"{path.name}: user {i} has no name")
        if role not in ROLES:
            raise ValueError(f"{path.name}: {name}'s role must be admin or member, not '{role}'")
        if not entry.get("password"):
            raise ValueError(f"{path.name}: {name} has no password (scribe users passwd {name})")
        if name.lower() in users:
            raise ValueError(f"{path.name}: two users are called {name}")
        users[name.lower()] = User(name, role, entry["password"])
    return users


def save_users(path: Path, users: list[User]) -> None:
    lines = ["# Who can sign in to the lore site. Manage it with `scribe users`;",
             "# passwords are stored as hashes. role: admin (can upload, delete and",
             "# re-index) or member (reads and asks).", ""]
    for u in users:
        lines += ["[[user]]", f"name = {json.dumps(u.name, ensure_ascii=False)}",
                  f"role = {json.dumps(u.role)}", f"password = {json.dumps(u.password)}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


class TooManyAttempts(Exception):
    pass


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


class Auth:
    def __init__(self, users_path: Path | None, secret_file: Path, session_days: int = 30):
        self.users_path = users_path
        self.secret_file = secret_file
        self.session_s = session_days * 86400
        self._users: tuple[float, dict[str, User]] | None = None
        self._secret: bytes | None = None
        self._failures: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    @property
    def users(self) -> dict[str, User]:
        """Re-read when the file changes, so `scribe users add` needs no restart."""
        path = self.users_path
        mtime = path.stat().st_mtime if path is not None and path.exists() else 0.0
        if self._users is None or self._users[0] != mtime:
            self._users = (mtime, load_users(path))
        return self._users[1]

    @property
    def enabled(self) -> bool:
        return bool(self.users)

    @property
    def secret(self) -> bytes:
        if self._secret is None:
            if os.environ.get(SECRET_ENV):
                self._secret = os.environ[SECRET_ENV].encode()
            else:
                if not self.secret_file.exists():
                    self.secret_file.parent.mkdir(parents=True, exist_ok=True)
                    self.secret_file.write_text(secrets.token_hex(32), encoding="utf-8")
                self._secret = self.secret_file.read_text(encoding="utf-8").strip().encode()
        return self._secret

    def login(self, name: str, password: str) -> User | None:
        key = name.strip().lower()
        now = time.time()
        with self._lock:
            recent = [t for t in self._failures.get(key, []) if now - t < FAILURE_WINDOW_S]
            self._failures[key] = recent
            if len(recent) >= MAX_FAILURES:
                raise TooManyAttempts(f"too many wrong passwords for {name}; try again in 15 minutes")
        user = self.users.get(key)
        # Hash something either way, so a wrong name takes as long as a wrong password.
        ok = check_password(password, user.password if user else hash_password(""))
        if user and ok:
            with self._lock:
                self._failures.pop(key, None)
            return user
        with self._lock:
            self._failures.setdefault(key, []).append(now)
        return None

    def _sign(self, payload: str) -> str:
        return _b64(hmac.new(self.secret, payload.encode(), hashlib.sha256).digest())

    def token(self, user: User) -> str:
        payload = f"{user.name.lower()}|{int(time.time()) + self.session_s}"
        return f"{_b64(payload.encode())}.{self._sign(payload + '|' + user.password[-16:])}"

    def user_for(self, token: str | None) -> User | None:
        if not token or "." not in token:
            return None
        body, signature = token.rsplit(".", 1)
        try:
            name, expires = _unb64(body).decode().rsplit("|", 1)
            if int(expires) < time.time():
                return None
        except (ValueError, UnicodeDecodeError):
            return None
        user = self.users.get(name)
        if user is None:
            return None
        expected = self._sign(f"{name}|{expires}|{user.password[-16:]}")
        return user if hmac.compare_digest(signature, expected) else None
