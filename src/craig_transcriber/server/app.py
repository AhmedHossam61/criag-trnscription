"""
The HTTP API behind the chat UI. Bound to 127.0.0.1 by default. To share it
through a tunnel, add logins first (scribe users add): with any user in
config/users.toml every route but the login page needs one, and only admins
can change the library. Without users, requests that come through a proxy
(an X-Forwarded-For header) are refused.

    GET    /                      the UI
    GET    /login, POST /login    sign in;  POST /logout
    GET    /api/status            index size, models, LLM quota
    GET    /api/documents         indexed files
    POST   /api/documents         upload files (multipart), then index them
    DELETE /api/documents?path=   delete an uploaded file, then re-index
    POST   /api/reindex           index new / changed files  {"rebuild": false}
    GET    /api/passages/{id}     one passage with its neighbours
    POST   /api/classify          {"question"} -> one-step or multi-step
    GET    /api/party             the characters in config/party.toml
    GET    /api/party/{id}        one character's page: sheet, sessions, mentions
    GET    /api/comics            the comics in assets/comics;  /api/comics/{id}/pages/{n}
    POST   /api/chat              {"question", "history", "mode"} -> server-sent events
"""

from __future__ import annotations

import html
import json
import re
import threading
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..rag.documents import DOCUMENT_TYPES
from ..rag.engine import RagEngine
from .auth import Auth, TooManyAttempts
from .comics import cast, load_comics

STATIC = Path(__file__).with_name("static")
EMBLEM_TYPES = (".png", ".svg", ".webp", ".jpg", ".jpeg")
COOKIE = "lore_session"
# What the login page itself needs before anyone has signed in.
OPEN_PATHS = {"/login", "/logout", "/static/app.css", "/emblem"}


class ChatMessage(BaseModel):
    role: str
    content: str = ""


class ChatRequest(BaseModel):
    question: str
    history: list[ChatMessage] = Field(default_factory=list)
    mode: str = "auto"


class ClassifyRequest(BaseModel):
    question: str


class ReindexRequest(BaseModel):
    rebuild: bool = False


def safe_name(name: str) -> str:
    """Keep a file name readable but harmless: no folders, no odd characters."""
    stem = Path(name or "upload").name
    stem = re.sub(r"[^\w.\- ()؀-ۿ]+", "_", stem).strip(" .") or "upload"
    return stem[:150]


def find_emblem(configured: Path | None) -> Path | None:
    """The emblem image, trying the usual extensions when none is given."""
    if configured is None:
        return None
    candidates = [configured] if configured.suffix.lower() in EMBLEM_TYPES else \
        [configured.with_name(configured.name + ext) for ext in EMBLEM_TYPES]
    return next((p for p in candidates if p.is_file()), None)


def create_app(engine: RagEngine, *, warm_up: bool = True) -> FastAPI:
    config = engine.config
    docs_dir = config.path(config.sources.documents)
    state = {"warmup_error": None}

    def _warm() -> None:
        try:
            engine.warm_up()
        except Exception as exc:  # noqa: BLE001 -- surfaced through /api/status
            state["warmup_error"] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()

    @asynccontextmanager
    async def lifespan(_app):
        if warm_up:
            threading.Thread(target=_warm, daemon=True).start()
        yield

    party = config.party.name or "The party"
    emblem = find_emblem(config.path(config.party.emblem))

    def branded(name: str) -> str:
        page = (STATIC / name).read_text(encoding="utf-8")
        page = (page.replace("{{party}}", html.escape(party))
                    .replace("{{tagline}}", html.escape(config.party.tagline)))
        if emblem:
            # Swap the built-in sigil for the party's emblem, and use it as the tab icon.
            start, end = page.index("<!--sigil-->"), page.index("<!--/sigil-->") + len("<!--/sigil-->")
            page = (page[:start] + '<img class="sigil" src="/emblem" alt="">' + page[end:])
            page = page.replace("<!--icon-->", '<link rel="icon" href="/emblem">')
        return page

    page, login_page = branded("index.html"), branded("login.html")
    auth = Auth(config.path(config.server.users), config.root / "data" / ".session_secret",
                config.server.session_days)

    app = FastAPI(title=f"{party} lore", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.middleware("http")
    async def require_login(request: Request, call_next):
        request.state.user = None
        try:
            enabled = auth.enabled
        except ValueError as exc:      # a mistake in config/users.toml: stay shut
            return JSONResponse({"detail": str(exc)}, 503)
        if not enabled:
            if "x-forwarded-for" in request.headers:
                return JSONResponse({"detail": "This site has no logins yet. On the server, run "
                                     "`scribe users add NAME --admin` before sharing it."}, 403)
            return await call_next(request)
        request.state.user = auth.user_for(request.cookies.get(COOKIE))
        if request.state.user or request.url.path in OPEN_PATHS:
            return await call_next(request)
        if request.url.path.startswith("/api/"):
            return JSONResponse({"detail": "sign in first"}, 401)
        return RedirectResponse("/login", 303)

    def require_admin(request: Request) -> None:
        user = request.state.user
        if auth.enabled and not (user and user.admin):
            raise HTTPException(403, "only an admin can change the library")

    def login_form(error: str = "", name: str = "", status: int = 200) -> HTMLResponse:
        body = (login_page.replace("{{error}}", html.escape(error))
                          .replace("{{name}}", html.escape(name))
                          .replace("{{error_hidden}}", "" if error else "hidden")
                          .replace("{{name_focus}}", "" if name else "autofocus")
                          .replace("{{password_focus}}", "autofocus" if name else ""))
        return HTMLResponse(body, status)

    @app.get("/", include_in_schema=False)
    def index_page():
        return HTMLResponse(page)

    @app.get("/login", include_in_schema=False)
    def login_get(request: Request):
        if not auth.enabled or request.state.user:
            return RedirectResponse("/", 303)
        return login_form()

    @app.post("/login", include_in_schema=False)
    def login_post(request: Request, name: str = Form(""), password: str = Form("")):
        if not auth.enabled:
            return RedirectResponse("/", 303)
        try:
            user = auth.login(name, password)
        except TooManyAttempts as exc:
            return login_form(str(exc).capitalize() + ".", name, 429)
        if user is None:
            return login_form("That name and password don't match.", name, 401)
        response = RedirectResponse("/", 303)
        https = request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"
        response.set_cookie(COOKIE, auth.token(user), max_age=auth.session_s, httponly=True,
                            samesite="lax", secure=https)
        return response

    @app.post("/logout", include_in_schema=False)
    def logout():
        response = RedirectResponse("/login", 303)
        response.delete_cookie(COOKIE)
        return response

    def _character(character_id: str):
        try:
            c = engine.party.get(character_id)
        except ValueError as exc:     # a mistake in config/party.toml
            raise HTTPException(422, str(exc)) from exc
        if c is None:
            raise HTTPException(404, f"no character '{character_id}' in the roster")
        return c

    @app.get("/api/party")
    def party_list():
        try:
            characters = engine.party.characters
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"name": party, "roster": config.party.roster,
                "characters": [engine.party.card(c) for c in characters]}

    @app.get("/api/party/{character_id}")
    def party_page(character_id: str):
        return engine.party.page(_character(character_id), engine.index.passages)

    @app.get("/api/party/{character_id}/portrait", include_in_schema=False)
    def party_portrait(character_id: str):
        path = engine.party.portrait(_character(character_id))
        if path is None:
            raise HTTPException(404, "no portrait")
        return FileResponse(path, headers={"Cache-Control": "no-cache"})

    def _comics():
        try:
            return load_comics(config.path(config.party.comics))
        except ValueError as exc:      # a mistake in a comic.toml
            raise HTTPException(422, str(exc)) from exc

    def _comic(comic) -> dict:
        try:
            roster = engine.party.characters
        except ValueError:
            roster = []
        return {"id": comic.id, "title": comic.title, "summary": comic.summary, "date": comic.date,
                "characters": cast(comic, roster),
                "pages": [f"/api/comics/{comic.id}/pages/{n}" for n in range(1, len(comic.pages) + 1)]}

    @app.get("/api/comics")
    def comics_list():
        return {"comics": [_comic(c) for c in _comics()]}

    @app.get("/api/comics/{comic_id}/pages/{number}", include_in_schema=False)
    def comic_page(comic_id: str, number: int):
        comic = next((c for c in _comics() if c.id == comic_id), None)
        if comic is None or not 1 <= number <= len(comic.pages):
            raise HTTPException(404, "no such page")
        return FileResponse(comic.pages[number - 1], headers={"Cache-Control": "no-cache"})

    @app.get("/emblem", include_in_schema=False)
    def emblem_file():
        if not emblem:
            raise HTTPException(404, "no emblem")
        return FileResponse(emblem, headers={"Cache-Control": "no-cache"})

    @app.get("/api/status")
    def status(request: Request):
        user = request.state.user
        return engine.status() | {"warmup_error": state["warmup_error"],
                                  "upload_types": sorted(DOCUMENT_TYPES), "emblem": emblem is not None,
                                  "login": auth.enabled,
                                  "user": {"name": user.name, "admin": user.admin} if user else None}

    @app.get("/api/documents")
    def documents():
        return {"documents": engine.documents(), "documents_dir": str(docs_dir)}

    def _reindex(rebuild: bool = False) -> dict:
        try:
            report = engine.reindex(rebuild=rebuild, progress=lambda _msg: None)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"summary": report.summary(), "added": report.added, "updated": report.updated,
                "removed": report.removed, "failed": report.failed, "passages": report.passages}

    @app.post("/api/documents")
    def upload(request: Request, files: list[UploadFile] = File(...)):
        require_admin(request)
        if docs_dir is None:
            raise HTTPException(400, "no documents folder configured in rag.toml")
        docs_dir.mkdir(parents=True, exist_ok=True)
        limit = config.server.max_upload_mb * 1024 * 1024
        saved, rejected = [], {}
        for upload_file in files:
            name = safe_name(upload_file.filename or "")
            if Path(name).suffix.lower() not in DOCUMENT_TYPES:
                rejected[name] = f"unsupported type (use {', '.join(sorted(DOCUMENT_TYPES))})"
                continue
            data = upload_file.file.read(limit + 1)
            if len(data) > limit:
                rejected[name] = f"larger than {config.server.max_upload_mb} MB"
                continue
            (docs_dir / name).write_bytes(data)
            saved.append(name)
        result = _reindex() if saved else {}
        return {"saved": saved, "rejected": rejected, **result}

    @app.delete("/api/documents")
    def delete_document(request: Request, path: str):
        require_admin(request)
        if docs_dir is None:
            raise HTTPException(400, "no documents folder configured")
        target = (config.root / path).resolve()
        if not target.is_relative_to(docs_dir.resolve()):
            raise HTTPException(403, "only files in the documents folder can be deleted here")
        if not target.is_file():
            raise HTTPException(404, f"{path} not found")
        target.unlink()
        return {"deleted": path, **_reindex()}

    @app.post("/api/reindex")
    def reindex(request: Request, req: ReindexRequest | None = None):
        require_admin(request)
        return _reindex(bool(req and req.rebuild))

    @app.get("/api/passages/{passage_id}")
    def passage(passage_id: str):
        found = engine.passage(passage_id)
        if found is None:
            raise HTTPException(404, "passage not found (the index may have changed)")
        return found

    @app.post("/api/classify")
    def classify(req: ClassifyRequest):
        return engine.classify(req.question).as_dict()

    @app.post("/api/chat")
    def chat(req: ChatRequest):
        if req.mode not in ("auto", "single", "multi"):
            raise HTTPException(400, "mode must be auto, single or multi")
        history = [m.model_dump() for m in req.history]

        def events():
            try:
                for event in engine.ask(req.question, history, req.mode):
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except Exception as exc:  # noqa: BLE001 -- report instead of cutting the stream
                traceback.print_exc()
                err = {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
                yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app
