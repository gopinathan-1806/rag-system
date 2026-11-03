"""Structured fit assessment of the resume against one job.

Unlike chat, this uses the *full* resume and posting (both fit comfortably in context): for a
whole-document judgement, retrieval can only lose information. Output is JSON validated by
pydantic, and each "matched skill" evidence quote is checked verbatim against the resume so the
model can't claim a skill it can't point to.
"""

import json
import logging

from pydantic import ValidationError

from app.errors import LLMUnavailableError, NotFoundError
from app.observability import LLM_ERRORS, LLM_TOKENS
from app.rag import guardrails, prompts
from app.rag.llm import LLM, Stopwatch, Usage
from app.schemas import FitReport, LLMFitOutput
from app.sessions import Session

log = logging.getLogger(__name__)


class FitService:
    def __init__(self, llm: LLM | None) -> None:
        self._llm = llm

    async def report(self, session: Session, job_id: str, *, refresh: bool = False) -> FitReport:
        job = session.jobs.get(job_id)
        if job is None or session.resume is None:
            raise NotFoundError("Upload a resume and select a valid job posting first.")
        if not refresh and job_id in session.reports:
            return session.reports[job_id]
        if self._llm is None:
            raise LLMUnavailableError(
                "No language model is configured. Set LLM_API_KEY (or LLM_BASE_URL) and restart."
            )

        messages = prompts.build_fit_messages(session.resume.text, job.label, job.text)
        usage = Usage()
        timer = Stopwatch()
        output: LLMFitOutput | None = None
        for attempt in (1, 2):  # one retry on malformed JSON
            raw = await self._complete(messages, usage)
            try:
                output = LLMFitOutput.model_validate(json.loads(raw))
                break
            except (json.JSONDecodeError, ValidationError):
                log.warning("fit_json_invalid", extra={"attempt": attempt})
        if output is None:
            raise LLMUnavailableError("The model returned an unreadable report. Please retry.")

        matched = [
            m.model_copy(update={"verified": guardrails.quote_in_document(m.evidence, session.resume.text)})
            for m in output.matched_skills
        ]
        report = FitReport(
            job_id=job.id,
            job_label=job.label,
            job_title=job.title,
            match_score=output.match_score,
            summary=output.summary,
            matched_skills=matched,
            missing_skills=output.missing_skills,
            experience_alignment=output.experience_alignment,
            interview_questions=output.interview_questions[:6],
        )
        session.reports[job_id] = report

        LLM_TOKENS.labels("prompt").inc(usage.prompt_tokens)
        LLM_TOKENS.labels("completion").inc(usage.completion_tokens)
        log.info(
            "fit_report",
            extra={
                "session": session.id[:6],
                "score": report.match_score,
                "unverified_evidence": sum(not m.verified for m in matched),
                "total_ms": timer.ms(),
                "prompt_tokens": usage.prompt_tokens,
            },
        )
        return report

    async def _complete(self, messages: list[dict[str, str]], usage: Usage) -> str:
        assert self._llm is not None  # noqa: S101  (guarded by report())
        try:
            return await self._llm.complete_json(messages, usage)
        except LLMUnavailableError:
            LLM_ERRORS.inc()
            raise
