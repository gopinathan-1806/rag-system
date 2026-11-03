"""HTTP routes. Thin: validation and status codes only, logic lives in services."""

import json
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, File, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.config import Settings
from app.errors import LLMUnavailableError, NotFoundError
from app.ingestion.service import IngestionService
from app.models import DocKind, Document
from app.observability import ACTIVE_SESSIONS
from app.rag.fit import FitService
from app.rag.pipeline import ChatService
from app.schemas import ChatIn, DocumentInfo, FitReport, SessionState, TextDocumentIn
from app.sessions import Session, SessionStore

router = APIRouter()


class Container:
    """Wired once at startup (see main.create_app) and stored on app.state."""

    def __init__(
        self, settings: Settings, sessions: SessionStore, ingestion: IngestionService,
        chat: ChatService, fit: FitService,
    ) -> None:  # fmt: skip
        self.settings = settings
        self.sessions = sessions
        self.ingestion = ingestion
        self.chat = chat
        self.fit = fit


def get_container(request: Request) -> Container:
    return request.app.state.container  # type: ignore[no-any-return]


C = Annotated[Container, Depends(get_container)]


def get_session(session_id: str, c: C) -> Session:
    return c.sessions.get(session_id)


S = Annotated[Session, Depends(get_session)]


def _info(doc: Document) -> DocumentInfo:
    return DocumentInfo(
        id=doc.id, kind=doc.kind, label=doc.label, title=doc.title, filename=doc.filename,
        chunks=len(doc.chunks), chars=len(doc.text), warnings=doc.warnings,
    )  # fmt: skip


def _state(session: Session, c: Container) -> SessionState:
    return SessionState(
        session_id=session.id,
        resume=_info(session.resume) if session.resume else None,
        jobs=[_info(j) for j in session.jobs.values()],
        llm_enabled=c.settings.llm_enabled,
    )


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(c: C) -> dict[str, object]:
    return {"status": "ready", "llm_configured": c.settings.llm_enabled}


@router.get("/metrics")
async def metrics(c: C) -> Response:
    ACTIVE_SESSIONS.set(len(c.sessions))
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.post("/api/sessions", status_code=201)
async def create_session(c: C) -> SessionState:
    return _state(c.sessions.create(), c)


@router.get("/api/sessions/{session_id}")
async def get_session_state(session: S, c: C) -> SessionState:
    return _state(session, c)


@router.delete("/api/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str, c: C) -> Response:
    c.sessions.delete(session_id)
    return Response(status_code=204)


async def _upload(kind: DocKind, session: Session, c: Container, file: UploadFile) -> DocumentInfo:
    # Read at most limit+1 bytes so an oversized upload can't exhaust memory.
    data = await file.read(c.settings.max_upload_bytes + 1)
    doc = await c.ingestion.add_file(session, kind, file.filename or "upload", data)
    return _info(doc)


@router.post("/api/sessions/{session_id}/resume", status_code=201)
async def upload_resume(session: S, c: C, file: Annotated[UploadFile, File()]) -> DocumentInfo:
    return await _upload("resume", session, c, file)


@router.post("/api/sessions/{session_id}/jobs", status_code=201)
async def upload_job(session: S, c: C, file: Annotated[UploadFile, File()]) -> DocumentInfo:
    return await _upload("job", session, c, file)


@router.post("/api/sessions/{session_id}/resume/text", status_code=201)
async def paste_resume(session: S, c: C, body: TextDocumentIn) -> DocumentInfo:
    return _info(await c.ingestion.add_text(session, "resume", body.title, body.text))


@router.post("/api/sessions/{session_id}/jobs/text", status_code=201)
async def paste_job(session: S, c: C, body: TextDocumentIn) -> DocumentInfo:
    return _info(await c.ingestion.add_text(session, "job", body.title, body.text))


_DEMO_FILES: list[tuple[DocKind, str]] = [
    ("resume", "resume_priya_raman.txt"),
    ("job", "job_ml_search_engineer.txt"),
    ("job", "job_staff_backend_engineer.txt"),
    ("job", "job_fullstack_engineer.txt"),
]


@router.post("/api/sessions/{session_id}/demo")
async def load_demo(session: S, c: C) -> SessionState:
    """Load the bundled sample resume + job postings so the app can be tried in one click."""
    samples = c.settings.samples_dir
    if not samples.is_dir():
        raise NotFoundError("Demo data is not available in this deployment.")
    for kind, name in _DEMO_FILES:
        if kind == "job" and any(j.filename == name for j in session.jobs.values()):
            continue
        await c.ingestion.add_file(session, kind, name, (samples / name).read_bytes())
    return _state(session, c)


@router.delete("/api/sessions/{session_id}/documents/{doc_id}", status_code=204)
async def delete_document(session: S, c: C, doc_id: str) -> Response:
    c.ingestion.remove(session, doc_id)
    return Response(status_code=204)


@router.post("/api/sessions/{session_id}/jobs/{job_id}/fit")
async def fit_report(session: S, c: C, job_id: str, refresh: bool = False) -> FitReport:
    return await c.fit.report(session, job_id, refresh=refresh)


@router.post("/api/sessions/{session_id}/chat")
async def chat(session: S, c: C, body: ChatIn) -> StreamingResponse:
    """Server-sent events. Pre-flight errors become normal HTTP errors; errors that happen
    after streaming started are delivered as an `error` event."""
    events = c.chat.answer(session, body.question)
    first = await anext(events)  # runs guardrails/retrieval so they can still fail with a 4xx

    async def stream() -> AsyncIterator[str]:
        try:
            yield _sse(first)
            async for event in events:
                yield _sse(event)
        except LLMUnavailableError as exc:
            yield _sse({"type": "error", "message": exc.message})

    return StreamingResponse(
        stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )  # fmt: skip


@router.delete("/api/sessions/{session_id}/chat", status_code=204)
async def clear_chat(session: S) -> Response:
    session.history.clear()
    return Response(status_code=204)


def _sse(event: dict[str, object]) -> str:
    return f"data: {json.dumps(event)}\n\n"
