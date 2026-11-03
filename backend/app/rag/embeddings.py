"""Embedding providers behind a tiny interface so tests can swap in a deterministic fake."""

import asyncio
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Vectors = NDArray[np.float32]


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> Vectors: ...
    def embed_query(self, text: str) -> Vectors: ...


class FastEmbedEmbedder:
    """Local ONNX embeddings (no network call, no data leaves the host, no per-token cost).

    BGE models are trained with an instruction prefix for *queries* (not for passages),
    which measurably improves question -> passage retrieval.
    """

    _QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding  # heavy import, keep it lazy

        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> Vectors:
        return self._normalize(np.array(list(self._model.embed(texts)), dtype=np.float32))

    def embed_query(self, text: str) -> Vectors:
        vec = np.array(list(self._model.embed([self._QUERY_PREFIX + text])), dtype=np.float32)
        first: Vectors = self._normalize(vec)[0]
        return first

    @staticmethod
    def _normalize(m: Vectors) -> Vectors:
        norms = np.linalg.norm(m, axis=-1, keepdims=True)
        out: Vectors = (m / np.clip(norms, 1e-12, None)).astype(np.float32)
        return out

    def warmup(self) -> None:
        self.embed_query("warmup")


async def aembed_documents(embedder: Embedder, texts: list[str]) -> Vectors:
    """ONNX inference is CPU-bound and synchronous; keep it off the event loop."""
    return await asyncio.to_thread(embedder.embed_documents, texts)


async def aembed_query(embedder: Embedder, text: str) -> Vectors:
    return await asyncio.to_thread(embedder.embed_query, text)
