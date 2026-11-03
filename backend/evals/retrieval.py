"""Retrieval quality eval using the REAL embedding model (downloads ~130 MB on first run).

    uv run python -m evals.retrieval

Each case is (question, expected substring that MUST appear in one of the top-k retrieved
chunks). It reports hit-rate@k and MRR for dense-only vs. hybrid, which is how the decision to
use hybrid retrieval was justified. Add cases here whenever a retrieval bug is reported.
"""

import sys
from pathlib import Path

from app.config import Settings
from app.ingestion.chunking import chunk_document, extract_title
from app.models import Document
from app.rag.embeddings import FastEmbedEmbedder
from app.rag.index import SessionIndex

SAMPLES = Path(__file__).resolve().parents[2] / "samples"
TOP_K = 3

# (question, file the answer lives in, substring expected in a retrieved chunk)
CASES = [
    ("Have I led a team?", "resume_priya_raman.txt", "Led a team of 4"),
    ("What cloud experience do I have?", "resume_priya_raman.txt", "AWS ECS"),
    ("Do I have vector database experience?", "resume_priya_raman.txt", "pgvector"),
    ("Any certifications?", "resume_priya_raman.txt", "AWS Certified"),
    ("How did I improve performance?", "resume_priya_raman.txt", "70%"),
    ("Do I know React?", "resume_priya_raman.txt", "React"),
    ("What does the ML role require for Kubernetes?", "job_ml_search_engineer.txt", "Kubernetes"),
    ("Which evaluation metrics does the search role mention?", "job_ml_search_engineer.txt", "NDCG"),
    ("How many years of experience for the staff role?", "job_staff_backend_engineer.txt", "8+ years"),
    ("Is fintech experience needed?", "job_staff_backend_engineer.txt", "fintech"),
    ("Does the full-stack job want healthcare knowledge?", "job_fullstack_engineer.txt", "HIPAA"),
    ("What is the nice to have for the search role?", "job_ml_search_engineer.txt", "MLflow"),
]  # fmt: skip


def build(embedder: FastEmbedEmbedder, settings: Settings) -> tuple[SessionIndex, dict[str, str]]:
    index = SessionIndex()
    doc_ids: dict[str, str] = {}
    for i, path in enumerate(sorted(SAMPLES.glob("*.txt"))):
        text = path.read_text()
        kind = "resume" if path.name.startswith("resume") else "job"
        doc = Document(
            id=f"d{i}",
            kind=kind,
            filename=path.name,
            title="",
            number=i,
            text=text,  # type: ignore[arg-type]
        )
        doc.title = extract_title(text, kind, path.name)  # type: ignore[arg-type]
        doc.chunks = chunk_document(doc, settings.chunk_max_chars)
        index.add(doc.chunks, embedder.embed_documents([c.embedding_text for c in doc.chunks]))
        doc_ids[path.name] = doc.id
    return index, doc_ids


def evaluate(hybrid: bool) -> tuple[float, float]:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    embedder = FastEmbedEmbedder(settings.embedding_model)
    index, doc_ids = build(embedder, settings)
    hits = 0
    rr = 0.0
    for question, fname, expected in CASES:
        qv = embedder.embed_query(question)
        query = question if hybrid else ""  # empty query => no lexical signal => dense only
        results = index.search(query, qv, top_k=TOP_K, doc_ids={doc_ids[fname]})
        rank = next((r for r, h in enumerate(results, 1) if expected.lower() in h.chunk.text.lower()), None)
        if rank:
            hits += 1
            rr += 1 / rank
        else:
            print(f"  MISS ({'hybrid' if hybrid else 'dense'}): {question!r}")
    n = len(CASES)
    return hits / n, rr / n


def main() -> int:
    for hybrid in (False, True):
        hit_rate, mrr = evaluate(hybrid)
        name = "hybrid (dense+BM25)" if hybrid else "dense only"
        print(f"{name:22s} hit@{TOP_K}={hit_rate:.2f}  MRR={mrr:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
