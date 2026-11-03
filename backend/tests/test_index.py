import numpy as np

from app.models import Chunk
from app.rag.index import SessionIndex, tokenize

from .conftest import HashingEmbedder


def _chunk(i: int, doc: str, text: str) -> Chunk:
    return Chunk(id=f"{doc}:{i}", doc_id=doc, doc_label=doc, doc_title="t", section="s", text=text)


def _index(*items: tuple[str, str]) -> tuple[SessionIndex, HashingEmbedder]:
    emb = HashingEmbedder()
    chunks = [_chunk(i, d, t) for i, (d, t) in enumerate(items)]
    idx = SessionIndex()
    idx.add(chunks, emb.embed_documents([c.embedding_text for c in chunks]))
    return idx, emb


def test_tokenize_keeps_tech_tokens() -> None:
    assert tokenize("C++ and C# with Node.js, CI/CD") == ["c++", "c#", "node.js", "ci", "cd"]


def test_search_ranks_relevant_first() -> None:
    idx, emb = _index(("r", "Python FastAPI services"), ("r", "Gardening and cooking hobbies"))
    hits = idx.search("python fastapi", emb.embed_query("python fastapi"), top_k=2)
    assert "Python" in hits[0].chunk.text


def test_lexical_match_rescues_rare_token() -> None:
    idx, emb = _index(("r", "general backend work"), ("r", "Certified CISSP professional"))
    hits = idx.search("CISSP", emb.embed_query("CISSP"), top_k=1)
    assert "CISSP" in hits[0].chunk.text


def test_doc_filter_and_removal() -> None:
    idx, emb = _index(("a", "python"), ("b", "python"))
    q = emb.embed_query("python")
    assert {h.chunk.doc_id for h in idx.search("python", q, top_k=5, doc_ids={"b"})} == {"b"}
    idx.remove_doc("a")
    assert len(idx) == 1
    assert [h.chunk.doc_id for h in idx.search("python", q, top_k=5)] == ["b"]
    idx.remove_doc("b")
    assert len(idx) == 0 and idx.search("python", q, top_k=5) == []


def test_empty_index_and_zero_k() -> None:
    idx = SessionIndex()
    assert idx.search("x", np.zeros(4, dtype=np.float32), top_k=3) == []
