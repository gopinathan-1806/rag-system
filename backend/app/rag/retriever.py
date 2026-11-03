"""Retrieval policy: which chunks go into the prompt for a given question.

The interesting part of this domain is that questions are *comparative*: "what am I missing
for Job #2" needs evidence from both the resume and one specific posting, and "which job fits
best" needs balanced evidence from every posting. A single top-k over the whole pool would let
one verbose posting crowd out the rest, so retrieval is done per document group with a budget.
"""

import re
from dataclasses import dataclass

from app.config import Settings
from app.models import Chunk
from app.rag.embeddings import Embedder, aembed_query
from app.rag.index import Hit
from app.sessions import Session

_JOB_REF = re.compile(r"\b(?:job|role|posting|position)\s*(?:#|number|no\.?)?\s*(\d{1,2})\b", re.I)


@dataclass(frozen=True)
class RetrievedPassage:
    source_id: str  # "S1", "S2", ... unique within one answer
    chunk: Chunk
    cosine: float


def referenced_job_numbers(question: str, session: Session) -> list[int]:
    """Job numbers explicitly mentioned in the question that exist in the session."""
    wanted = {int(m) for m in _JOB_REF.findall(question)}
    return sorted(n for n in wanted if session.job_by_number(n) is not None)


async def retrieve(
    question: str, session: Session, embedder: Embedder, settings: Settings
) -> list[RetrievedPassage]:
    qvec = await aembed_query(embedder, question)
    hits: list[Hit] = []

    if session.resume:
        hits += session.index.search(question, qvec, top_k=settings.resume_top_k, doc_ids={session.resume.id})

    jobs = list(session.jobs.values())
    mentioned = referenced_job_numbers(question, session)
    if mentioned:
        jobs = [j for j in jobs if j.number in mentioned]
    if jobs:
        # Equal share per posting (min 2) so one long posting can't starve the others.
        per_job = max(2, settings.jobs_total_top_k // len(jobs))
        for job in jobs:
            hits += session.index.search(question, qvec, top_k=per_job, doc_ids={job.id})

    return _apply_budget(hits, settings.max_context_chars)


def _apply_budget(hits: list[Hit], max_chars: int) -> list[RetrievedPassage]:
    """Keep passages in document order (resume first, then jobs, then reading order) within
    the character budget, dropping the lowest-ranked first."""
    kept: list[Hit] = []
    used = 0
    for hit in sorted(hits, key=lambda h: -h.score):
        cost = len(hit.chunk.text) + 80
        if used + cost > max_chars:
            continue
        kept.append(hit)
        used += cost

    def reading_order(h: Hit) -> tuple[int, str, int]:
        is_job = 0 if h.chunk.doc_label == "Resume" else 1
        _, _, idx = h.chunk.id.rpartition(":")
        return (is_job, h.chunk.doc_label, int(idx))

    kept.sort(key=reading_order)
    return [
        RetrievedPassage(source_id=f"S{i}", chunk=h.chunk, cosine=h.cosine)
        for i, h in enumerate(kept, start=1)
    ]


def format_passages(passages: list[RetrievedPassage]) -> str:
    return "\n\n".join(
        f"[{p.source_id}] {p.chunk.doc_label} - {p.chunk.doc_title} - {p.chunk.section}\n{p.chunk.text}"
        for p in passages
    )
