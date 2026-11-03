"""App factory, middleware, and the process entrypoint."""

import logging
import socket
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app.api import Container, router
from app.config import Settings, get_settings
from app.errors import AppError
from app.ingestion.service import IngestionService
from app.observability import REQUEST_LATENCY, REQUESTS, configure_logging
from app.rag.embeddings import Embedder, FastEmbedEmbedder
from app.rag.fit import FitService
from app.rag.llm import LLM, OpenAICompatibleLLM
from app.rag.pipeline import ChatService
from app.sessions import SessionStore

log = logging.getLogger("app")

_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
    "Content-Security-Policy": (
        "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'none'"
    ),
}


def create_app(
    settings: Settings | None = None, *, embedder: Embedder | None = None, llm: LLM | None = None
) -> FastAPI:
    """`embedder` / `llm` are injectable so tests run offline and deterministically."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        nonlocal embedder, llm
        configure_logging(settings.log_level)
        if embedder is None:
            log.info("loading_embedding_model", extra={"model": settings.embedding_model})
            fe = FastEmbedEmbedder(settings.embedding_model)
            fe.warmup()  # download/load now, not on the first user request
            embedder = fe
        if llm is None and settings.llm_enabled:
            llm = OpenAICompatibleLLM(settings)
        if llm is None:
            log.warning("llm_not_configured", extra={"hint": "set LLM_API_KEY"})
        app.state.container = Container(
            settings=settings,
            sessions=SessionStore(settings.session_ttl_s, settings.max_sessions),
            ingestion=IngestionService(settings, embedder),
            chat=ChatService(settings, embedder, llm),
            fit=FitService(llm),
        )
        yield

    app = FastAPI(title="Career Copilot", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def observe(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        request_id = uuid.uuid4().hex[:12]
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
        finally:
            route = getattr(request.scope.get("route"), "path", "unmatched")
            elapsed = time.perf_counter() - start
            REQUESTS.labels(request.method, route, str(status)).inc()
            REQUEST_LATENCY.labels(route).observe(elapsed)
            if route not in ("/healthz", "/metrics"):
                log.info(
                    "request",
                    extra={"request_id": request_id, "method": request.method, "route": route,
                           "status": status, "ms": round(elapsed * 1000, 1)},
                )  # fmt: skip
        response.headers["X-Request-ID"] = request_id
        for k, v in _SECURITY_HEADERS.items():
            response.headers.setdefault(k, v)
        return response

    @app.exception_handler(AppError)
    async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"error": exc.message})

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception) -> JSONResponse:
        log.error("unhandled_error", exc_info=exc)  # details server-side only
        return JSONResponse(status_code=500, content={"error": "Something went wrong."})

    app.include_router(router)

    static = settings.static_dir
    if static and Path(static).is_dir():
        app.mount("/", StaticFiles(directory=static, html=True), name="ui")
    return app


def _bind_host(settings: Settings) -> str:
    """Never bind 0.0.0.0. In a container, loopback is unreachable from published ports, so
    bind to the container's own interface address instead."""
    if settings.host == "container":
        return socket.gethostbyname(socket.gethostname())
    return settings.host


def run() -> None:
    settings = get_settings()
    uvicorn.run(
        "app.main:app", host=_bind_host(settings), port=settings.port, log_config=None,
        factory=False,
    )  # fmt: skip


def _default_app() -> FastAPI:
    return create_app()


app = _default_app()

if __name__ == "__main__":
    run()
