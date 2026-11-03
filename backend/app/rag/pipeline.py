"""Chat orchestration: guardrail -> retrieve -> prompt -> stream -> verify citations.

Plain async code instead of LangChain/LlamaIndex: the flow is a straight line of five steps,
and owning it makes every step observable, testable and easy to explain.
"""

import logging
from collections.abc import AsyncIterator
from typing import Any

from app.config import Settings
from app.errors import GuardrailViolation, LLMUnavailableError
from app.models import Turn
from app.observability import (
    CITATION_CHECKS,
    GUARDRAIL_BLOCKS,
    LLM_ERRORS,
    LLM_FIRST_TOKEN,
    LLM_TOKENS,
    RETRIEVAL_LATENCY,
)
from app.rag import guardrails, prompts
from app.rag.embeddings import Embedder
from app.rag.llm import LLM, Stopwatch, Usage
from app.rag.retriever import RetrievedPassage, format_passages, retrieve
from app.schemas import Source
from app.sessions import Session

log = logging.getLogger(__name__)

Event = dict[str, Any]


class ChatService:
    def __init__(self, settings: Settings, embedder: Embedder, llm: LLM | None) -> None:
        self._s = settings
        self._embedder = embedder
        self._llm = llm

    async def answer(self, session: Session, question: str) -> AsyncIterator[Event]:
        """Yields events: sources -> token* -> done. Errors are raised before the first yield
        when possible (so the HTTP layer can return a proper status code)."""
        try:
            question = guardrails.check_question(question, self._s.max_question_chars)
        except GuardrailViolation as v:
            GUARDRAIL_BLOCKS.labels(v.rule).inc()
            log.warning("guardrail_blocked", extra={"rule": v.rule, "session": session.id[:6]})
            raise

        if session.resume is None or not session.jobs:
            yield {"type": "token", "text": prompts.NO_DOCUMENTS}
            yield {"type": "done", "cited": [], "grounded": False, "answer": prompts.NO_DOCUMENTS}
            return
        if self._llm is None:
            raise LLMUnavailableError(
                "No language model is configured. Set LLM_API_KEY (or LLM_BASE_URL) and restart."
            )

        total = Stopwatch()
        retrieval_timer = Stopwatch()
        passages = await retrieve(question, session, self._embedder, self._s)
        retrieval_ms = retrieval_timer.ms()
        RETRIEVAL_LATENCY.observe(retrieval_ms / 1000)

        yield {"type": "sources", "sources": [_to_source(p).model_dump() for p in passages]}

        messages = prompts.build_chat_messages(
            question,
            format_passages(passages),
            session.history[-self._s.max_history_turns :],
            summary_of_docs=_describe_documents(session),
        )

        usage = Usage()
        parts: list[str] = []
        llm_timer = Stopwatch()
        first_token_ms: float | None = None
        try:
            async for delta in self._llm.stream(messages, usage):
                if first_token_ms is None:
                    first_token_ms = llm_timer.ms()
                    LLM_FIRST_TOKEN.observe(first_token_ms / 1000)
                parts.append(delta)
                yield {"type": "token", "text": delta}
        except LLMUnavailableError:
            LLM_ERRORS.inc()
            log.exception("llm_failed", extra={"session": session.id[:6]})
            raise

        answer = "".join(parts)
        check = guardrails.check_citations(answer, {p.source_id for p in passages})
        if check.invalid:
            answer = guardrails.strip_invalid_citations(answer, check.invalid)
            CITATION_CHECKS.labels("invalid_removed").inc()
        elif check.cited:
            CITATION_CHECKS.labels("ok").inc()
        else:
            CITATION_CHECKS.labels("none").inc()

        session.history.append(Turn(question=question, answer=answer))
        del session.history[: -self._s.max_history_turns]

        LLM_TOKENS.labels("prompt").inc(usage.prompt_tokens)
        LLM_TOKENS.labels("completion").inc(usage.completion_tokens)
        log.info(
            "chat_answered",
            extra={
                "session": session.id[:6],
                "model": self._llm.model,
                "prompt_version": prompts.PROMPT_VERSION,
                "passages": len(passages),
                "cited": len(check.cited),
                "invalid_citations": len(check.invalid),
                "retrieval_ms": retrieval_ms,
                "first_token_ms": first_token_ms,
                "total_ms": total.ms(),
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
            },
        )
        yield {
            "type": "done",
            "answer": answer,  # authoritative text (invalid citations removed)
            "cited": check.cited,
            "grounded": bool(check.cited),
            "stats": {"retrieval_ms": retrieval_ms, "total_ms": total.ms()},
        }


def _to_source(p: RetrievedPassage) -> Source:
    c = p.chunk
    return Source(
        id=p.source_id, label=c.doc_label, title=c.doc_title, section=c.section,
        text=c.text, cosine=round(p.cosine, 3),
    )  # fmt: skip


def _describe_documents(session: Session) -> str:
    parts = []
    if session.resume:
        parts.append(f"Resume ({session.resume.title})")
    parts += [f"{j.label} ({j.title})" for j in session.jobs.values()]
    return "; ".join(parts)
