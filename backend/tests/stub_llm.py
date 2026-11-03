"""Minimal OpenAI-compatible stub server (chat completions, streaming + JSON mode).

Used (1) by tests to exercise the real OpenAI client code path over HTTP, and (2) to run the UI
without an API key:  uv run uvicorn tests.stub_llm:app --port 8020
then start the app with  LLM_BASE_URL=http://127.0.0.1:8020/v1  LLM_MODEL=stub

It does NOT understand language: it quotes the retrieved passages it is given. Screenshots made
with it demonstrate the UI and the grounding plumbing, not answer quality.
"""

import json
import re
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()
_PASSAGE = re.compile(r"^\[(S\d+)\] (Resume|Job #\d+) - (.+?) - (.+)\n(.+)$", re.MULTILINE)


def _chat_answer(prompt: str) -> str:
    passages = _PASSAGE.findall(prompt)
    if not passages:
        return "I could not find that in the provided documents."
    resume = [p for p in passages if p[1] == "Resume"]
    jobs = [p for p in passages if p[1] != "Resume"]
    lines = ["**Here is what the retrieved passages say** (stub model, quoting sources):", ""]
    for sid, label, _title, section, text in jobs[:2]:
        lines.append(f"- **{label} · {section}:** {text.strip()[:140]} [{sid}]")
    for sid, _label, _title, section, text in resume[:2]:
        lines.append(f"- **Your resume · {section}:** {text.strip()[:140]} [{sid}]")
    return "\n".join(lines)


def _fit_json(prompt: str) -> dict[str, Any]:
    resume = re.search(r"<resume>\n(.*?)\n</resume>", prompt, re.S)
    first_bullet = next(
        (ln[2:] for ln in (resume.group(1) if resume else "").splitlines() if ln.startswith("- ")), ""
    )
    return {
        "match_score": 68,
        "summary": "Stub report: shows the UI shape, not a real assessment.",
        "matched_skills": [{"skill": "Example skill", "evidence": first_bullet[:120]}],
        "missing_skills": [
            {"skill": "Example gap", "importance": "required", "suggestion": "Stub suggestion."}
        ],
        "experience_alignment": "Stub alignment text.",
        "interview_questions": [{"question": "Stub question?", "talking_point": "Stub talking point."}] * 4,
    }


@app.post("/v1/chat/completions", response_model=None)
async def completions(request: Request) -> StreamingResponse | JSONResponse:
    body = await request.json()
    prompt = body["messages"][-1]["content"]
    created = int(time.time())

    if body.get("stream"):

        async def gen() -> AsyncIterator[str]:
            def chunk(delta: dict[str, str], usage: dict[str, int] | None = None) -> str:
                payload: dict[str, Any] = {
                    "id": "stub", "object": "chat.completion.chunk", "created": created, "model": "stub",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}] if usage is None else [],
                }  # fmt: skip
                if usage:
                    payload["usage"] = usage
                return f"data: {json.dumps(payload)}\n\n"

            for token in re.findall(r"\S+\s*", _chat_answer(prompt)):
                yield chunk({"content": token})
            yield chunk({}, {"prompt_tokens": 321, "completion_tokens": 45, "total_tokens": 366})
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    content = json.dumps(_fit_json(prompt))
    return JSONResponse(
        {
            "id": "stub",
            "object": "chat.completion",
            "created": created,
            "model": "stub",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 500, "completion_tokens": 120, "total_tokens": 620},
        }  # fmt: skip
    )
