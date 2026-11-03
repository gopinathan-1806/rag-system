"""Exercise the real OpenAI client wrapper against a local OpenAI-compatible stub over HTTP."""

import httpx
import pytest
from openai import AsyncOpenAI

from app.config import Settings
from app.errors import LLMUnavailableError
from app.rag.llm import OpenAICompatibleLLM, Usage

from .stub_llm import app as stub_app


def _llm(handler: httpx.AsyncBaseTransport | None = None) -> OpenAICompatibleLLM:
    llm = OpenAICompatibleLLM(Settings(llm_api_key="k", _env_file=None))  # type: ignore[call-arg,arg-type]
    llm._client = AsyncOpenAI(
        api_key="k",
        base_url="http://stub/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=handler or httpx.ASGITransport(app=stub_app)),
    )
    return llm


MSGS = [
    {"role": "system", "content": "s"},
    {"role": "user", "content": "<passages>\n[S1] Resume - Priya - Skills\nPython and FastAPI\n</passages>"},
]


async def test_stream_yields_tokens_and_usage() -> None:
    usage = Usage()
    text = "".join([t async for t in _llm().stream(MSGS, usage)])
    assert "[S1]" in text and usage.prompt_tokens == 321 and usage.completion_tokens == 45


async def test_complete_json_returns_parseable_json() -> None:
    usage = Usage()
    raw = await _llm().complete_json([{"role": "user", "content": "<resume>\n- did x\n</resume>"}], usage)
    assert '"match_score"' in raw and usage.prompt_tokens == 500


@pytest.mark.parametrize(
    ("status", "fragment"), [(401, "credentials"), (429, "rate limited"), (500, "unavailable")]
)
async def test_provider_errors_become_safe_messages(status: int, fragment: str) -> None:
    def boom(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": "secret internal detail sk-123"}})

    llm = _llm(httpx.MockTransport(boom))
    with pytest.raises(LLMUnavailableError) as e:
        _ = [t async for t in llm.stream(MSGS, Usage())]
    assert fragment in e.value.message and "sk-123" not in e.value.message
    with pytest.raises(LLMUnavailableError):
        await llm.complete_json(MSGS, Usage())
