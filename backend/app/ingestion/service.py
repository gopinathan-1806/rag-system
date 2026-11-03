"""Ingestion service: parse -> chunk -> embed -> index, with per-session limits."""

import logging
import uuid

from app.config import Settings
from app.errors import LimitError, ParseError
from app.ingestion.chunking import chunk_document, extract_title
from app.ingestion.parsing import normalize_text, parse_document
from app.models import DocKind, Document
from app.observability import INGESTED
from app.rag.embeddings import Embedder, aembed_documents
from app.rag.guardrails import scan_document_for_injection
from app.sessions import Session

log = logging.getLogger(__name__)


class IngestionService:
    def __init__(self, settings: Settings, embedder: Embedder) -> None:
        self._s = settings
        self._embedder = embedder

    async def add_file(self, session: Session, kind: DocKind, filename: str, data: bytes) -> Document:
        text = parse_document(
            filename, data, max_bytes=self._s.max_upload_bytes, max_chars=self._s.max_document_chars
        )
        return await self._add(session, kind, filename, text, title=None)

    async def add_text(self, session: Session, kind: DocKind, title: str, text: str) -> Document:
        text = normalize_text(text)
        if len(text) > self._s.max_document_chars:
            raise ParseError(f"Text is too long (limit {self._s.max_document_chars:,} characters).")
        return await self._add(session, kind, f"{title or 'pasted'}.txt", text, title=title or None)

    async def _add(
        self, session: Session, kind: DocKind, filename: str, text: str, title: str | None
    ) -> Document:
        if len(text) < self._s.min_document_chars:
            raise ParseError(
                "This document has very little text. If it is a scanned PDF, please upload "
                "a text-based version (OCR is not supported)."
            )
        if kind == "job" and len(session.jobs) >= self._s.max_jobs_per_session:
            raise LimitError(f"You can compare up to {self._s.max_jobs_per_session} job postings.")

        doc = Document(
            id=uuid.uuid4().hex[:12],
            kind=kind,
            filename=filename,
            title=title or extract_title(text, kind, filename),
            number=session.peek_job_number() if kind == "job" else None,
            text=text,
            warnings=scan_document_for_injection(text),
        )
        doc.chunks = chunk_document(doc, self._s.chunk_max_chars)
        if not doc.chunks:
            raise ParseError("Could not find any usable content in this document.")

        vectors = await aembed_documents(self._embedder, [c.embedding_text for c in doc.chunks])

        # Commit only after everything succeeded, so a failure leaves the session untouched.
        if kind == "resume":
            if session.resume:
                self.remove(session, session.resume.id)
            session.resume = doc
            session.reports.clear()  # reports depended on the old resume
        else:
            session.consume_job_number()  # only on success, so failed uploads don't burn numbers
            session.jobs[doc.id] = doc
        session.index.add(doc.chunks, vectors)

        INGESTED.labels(kind).inc()
        log.info(
            "document_ingested",
            extra={"kind": kind, "chunks": len(doc.chunks), "chars": len(text), "session": session.id[:6]},
        )
        return doc

    def remove(self, session: Session, doc_id: str) -> None:
        if session.resume and session.resume.id == doc_id:
            session.resume = None
            session.reports.clear()
        else:
            session.jobs.pop(doc_id, None)
            session.reports.pop(doc_id, None)
        session.index.remove_doc(doc_id)
