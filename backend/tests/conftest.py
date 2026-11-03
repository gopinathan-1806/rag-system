"""Test doubles: deterministic offline embedder and a scriptable LLM."""

import hashlib
import json
import re
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.rag.embeddings import Vectors
from app.rag.index import tokenize
from app.rag.llm import Message, Usage

SAMPLES = Path(__file__).resolve().parents[2] / "samples"


class HashingEmbedder:
    """Bag-of-words hashed into 256 dims. Not semantic, but similar wording => similar vectors,
    which is all the plumbing tests need."""

    dim = 256

    def _vec(self, text: str) -> Vectors:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            h = int(hashlib.md5(tok.encode(), usedforsecurity=False).hexdigest(), 16)
            v[h % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts: list[str]) -> Vectors:
        return np.vstack([self._vec(t) for t in texts])

    def embed_query(self, text: str) -> Vectors:
        return self._vec(text)


class FakeLLM:
    model = "fake-model"

    def __init__(self) -> None:
        self.answer = "You know Python [S1]."
        self.fit_payload: dict[str, object] | str = {}
        self.calls: list[list[Message]] = []

    async def stream(self, messages: list[Message], usage: Usage) -> AsyncIterator[str]:
        self.calls.append(messages)
        usage.prompt_tokens, usage.completion_tokens = 100, 10
        for word in re.findall(r"\S+\s*", self.answer):
            yield word

    async def complete_json(self, messages: list[Message], usage: Usage) -> str:
        self.calls.append(messages)
        p = self.fit_payload
        return p if isinstance(p, str) else json.dumps(p)


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def settings() -> Settings:
    return Settings(static_dir=None, llm_api_key=None, samples_dir=SAMPLES, _env_file=None)  # type: ignore[call-arg]


@pytest.fixture
def client(settings: Settings, embedder: HashingEmbedder, llm: FakeLLM):  # type: ignore[no-untyped-def]
    with TestClient(create_app(settings, embedder=embedder, llm=llm)) as c:
        yield c


def sample(name: str) -> bytes:
    return (SAMPLES / name).read_bytes()
