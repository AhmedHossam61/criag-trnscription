"""
The HTTP API behind the chat UI. Bound to 127.0.0.1 by default: it can read
and delete files in data/documents, so it is not meant to face a network.

    GET    /                      the UI
    GET    /api/status            index size, models, LLM quota
    GET    /api/documents         indexed files
    POST   /api/documents         upload files (multipart), then index them
    DELETE /api/documents?path=   delete an uploaded file, then re-index
    POST   /api/reindex           index new / changed files  {"rebuild": false}
    GET    /api/passages/{id}     one passage with its neighbours
    POST   /api/classify          {"question"} -> one-step or multi-step
    POST   /api/chat              {"question", "history", "mode"} -> server-sent events
"""

from __future__ import annotations

import json
import re
import threading
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..rag.documents import DOCUMENT_TYPES
from ..rag.engine import RagEngine

STATIC = Path(__file__).with_name("static")


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

    app = FastAPI(title="Craig Transcriber archive", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def index_page():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/status")
    def status():
        return engine.status() | {"warmup_error": state["warmup_error"],
                                  "upload_types": sorted(DOCUMENT_TYPES)}

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
    def upload(files: list[UploadFile] = File(...)):
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
    def delete_document(path: str):
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
    def reindex(req: ReindexRequest | None = None):
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
