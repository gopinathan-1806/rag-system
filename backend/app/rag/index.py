"""In-memory hybrid index (dense cosine + BM25) for one session.

Corpus size is tiny (a resume plus <=5 postings = a few hundred chunks), so a numpy matrix
and brute-force search is exact, dependency-free and faster than any network round trip to a
vector DB. See README "Vector database" for when and how this would be swapped out.

Dense retrieval alone is weak at exact-token matches that dominate this domain ("Kubernetes",
"PyTorch", "CISSP"), so scores from BM25 and cosine are fused with Reciprocal Rank Fusion.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from app.models import Chunk
from app.rag.embeddings import Vectors

_TOKEN = re.compile(r"[a-z0-9][a-z0-9+#.\-]*[a-z0-9+#]|[a-z0-9]")
_STOP = frozenset(
    "a an and are as at be by for from has have in is it of on or that the to with you your we our "
    "will this these those their can may not but if than then so such into over about".split()
)
_RRF_K = 60


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # fused RRF score (relative ranking only)
    cosine: float  # raw semantic similarity, used for the "is this relevant at all" check


class SessionIndex:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._matrix: Vectors = np.empty((0, 0), dtype=np.float32)
        self._tf: list[Counter[str]] = []
        self._df: Counter[str] = Counter()
        self._avg_len = 0.0

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunks: list[Chunk], vectors: Vectors) -> None:
        if not chunks:
            return
        self._chunks.extend(chunks)
        self._matrix = vectors if self._matrix.size == 0 else np.vstack([self._matrix, vectors])
        for c in chunks:
            tf = Counter(tokenize(c.embedding_text))
            self._tf.append(tf)
            self._df.update(tf.keys())
        self._avg_len = sum(sum(tf.values()) for tf in self._tf) / len(self._tf)

    def remove_doc(self, doc_id: str) -> None:
        keep = [i for i, c in enumerate(self._chunks) if c.doc_id != doc_id]
        if len(keep) == len(self._chunks):
            return
        self._chunks = [self._chunks[i] for i in keep]
        self._tf = [self._tf[i] for i in keep]
        self._matrix = self._matrix[keep] if keep else np.empty((0, 0), dtype=np.float32)
        self._df = Counter()
        for tf in self._tf:
            self._df.update(tf.keys())
        total = sum(sum(tf.values()) for tf in self._tf)
        self._avg_len = total / len(self._tf) if self._tf else 0.0

    def search(
        self, query: str, query_vec: Vectors, *, top_k: int, doc_ids: set[str] | None = None
    ) -> list[Hit]:
        """Top-k chunks for the query, optionally restricted to a set of documents."""
        idx = [i for i, c in enumerate(self._chunks) if doc_ids is None or c.doc_id in doc_ids]
        if not idx or top_k <= 0:
            return []

        cosines = self._matrix[idx] @ query_vec
        q_tokens = tokenize(query)
        bm25 = np.array([self._bm25(q_tokens, i) for i in idx])

        dense_rank = {pos: r for r, pos in enumerate(np.argsort(-cosines))}
        # BM25 == 0 means "no lexical overlap": give it no rank contribution at all.
        lex_order = [pos for pos in np.argsort(-bm25) if bm25[pos] > 0]
        lex_rank = {pos: r for r, pos in enumerate(lex_order)}

        fused: list[tuple[float, int]] = []
        for pos in range(len(idx)):
            s = 1.0 / (_RRF_K + dense_rank[pos])
            if pos in lex_rank:
                s += 1.0 / (_RRF_K + lex_rank[pos])
            fused.append((s, pos))
        fused.sort(key=lambda t: -t[0])

        return [
            Hit(chunk=self._chunks[idx[pos]], score=s, cosine=float(cosines[pos])) for s, pos in fused[:top_k]
        ]

    def _bm25(self, q_tokens: list[str], i: int, k1: float = 1.5, b: float = 0.75) -> float:
        tf = self._tf[i]
        n = len(self._tf)
        dl = sum(tf.values())
        score = 0.0
        for t in set(q_tokens):
            f = tf.get(t, 0)
            if not f:
                continue
            idf = math.log(1 + (n - self._df[t] + 0.5) / (self._df[t] + 0.5))
            score += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / (self._avg_len or 1)))
        return score
