"""LLM access behind a small protocol. Works with any OpenAI-compatible endpoint."""

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

from app.config import Settings
from app.errors import LLMUnavailableError

Message = dict[str, str]


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLM(Protocol):
    model: str

    def stream(self, messages: list[Message], usage: Usage) -> AsyncIterator[str]: ...
    async def complete_json(self, messages: list[Message], usage: Usage) -> str: ...


class OpenAICompatibleLLM:
    def __init__(self, settings: Settings) -> None:
        key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else "not-needed"
        self._client = AsyncOpenAI(
            api_key=key,
            base_url=settings.llm_base_url,
            timeout=settings.llm_timeout_s,
            max_retries=2,
        )
        self.model = settings.llm_model
        self._temperature = settings.llm_temperature

    async def stream(self, messages: list[Message], usage: Usage) -> AsyncIterator[str]:
        try:
            response = await self._client.chat.completions.create(  # type: ignore[call-overload]
                model=self.model,
                messages=messages,
                temperature=self._temperature,
                stream=True,
                stream_options={"include_usage": True},
            )
            async for event in response:
                if event.usage:
                    usage.prompt_tokens = event.usage.prompt_tokens
                    usage.completion_tokens = event.usage.completion_tokens
                if event.choices and (delta := event.choices[0].delta.content):
                    yield delta
        except (APIConnectionError, APITimeoutError, APIStatusError) as exc:
            raise LLMUnavailableError(_friendly(exc)) from exc

    async def complete_json(self, messages: list[Message], usage: Usage) -> str:
        try:
            response = await self._client.chat.completions.create(  # type: ignore[call-overload]
                model=self.model,
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"},
            )
        except (APIConnectionError, APITimeoutError, APIStatusError) as exc:
            raise LLMUnavailableError(_friendly(exc)) from exc
        if response.usage:
            usage.prompt_tokens += response.usage.prompt_tokens
            usage.completion_tokens += response.usage.completion_tokens
        return response.choices[0].message.content or "{}"


def _friendly(exc: Exception) -> str:
    """Never leak provider response bodies (can contain request details) to the browser."""
    if isinstance(exc, APIStatusError) and exc.status_code in (401, 403):
        return "The language model rejected the configured credentials."
    if isinstance(exc, APIStatusError) and exc.status_code == 429:
        return "The language model is rate limited. Please retry in a moment."
    return "The language model is currently unavailable. Please retry."


class Stopwatch:
    def __init__(self) -> None:
        self._t0 = time.perf_counter()

    def ms(self) -> float:
        return round((time.perf_counter() - self._t0) * 1000, 1)
