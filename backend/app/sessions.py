"""Ephemeral in-memory session store.

Resumes are PII, so sessions are intentionally never persisted: a TTL evicts them and a
DELETE endpoint lets users wipe them early. For multi-instance deployments this moves to
Redis (TTL) + a shared vector store; see README "Productionizing".
"""

import secrets
import time
from dataclasses import dataclass, field

from app.errors import LimitError, NotFoundError
from app.models import Document, Turn
from app.rag.index import SessionIndex
from app.schemas import FitReport


@dataclass
class Session:
    id: str
    index: SessionIndex = field(default_factory=SessionIndex)
    resume: Document | None = None
    jobs: dict[str, Document] = field(default_factory=dict)
    history: list[Turn] = field(default_factory=list)
    reports: dict[str, FitReport] = field(default_factory=dict)  # keyed by job doc id
    last_used: float = field(default_factory=time.monotonic)
    _next_job_number: int = 1

    def documents(self) -> list[Document]:
        return ([self.resume] if self.resume else []) + list(self.jobs.values())

    def peek_job_number(self) -> int:
        return self._next_job_number

    def consume_job_number(self) -> None:
        self._next_job_number += 1

    def job_by_number(self, number: int) -> Document | None:
        return next((j for j in self.jobs.values() if j.number == number), None)


class SessionStore:
    def __init__(self, ttl_s: int, max_sessions: int) -> None:
        self._ttl = ttl_s
        self._max = max_sessions
        self._sessions: dict[str, Session] = {}

    def create(self) -> Session:
        self._evict_expired()
        if len(self._sessions) >= self._max:
            raise LimitError("The service is at capacity. Please try again later.")
        session = Session(id=secrets.token_urlsafe(24))  # unguessable: it is the only credential
        self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Session:
        self._evict_expired()
        session = self._sessions.get(session_id)
        if session is None:
            raise NotFoundError("Session not found or expired. Please start a new one.")
        session.last_used = time.monotonic()
        return session

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def __len__(self) -> int:
        return len(self._sessions)

    def _evict_expired(self) -> None:
        cutoff = time.monotonic() - self._ttl
        for sid in [s for s, v in self._sessions.items() if v.last_used < cutoff]:
            del self._sessions[sid]
