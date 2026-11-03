# Career Copilot

Upload a resume and up to five job postings, then ask questions about fit, skill gaps, experience
alignment and interview prep. Every claim in an answer is cited back to the passage it came from.

![Chat with citations](screenshots/03-chat-with-citation.png)

| | |
|---|---|
| ![Empty](screenshots/01-empty.png) | ![Docs loaded](screenshots/02-documents-loaded.png) |
| ![Guardrail](screenshots/04-guardrail-block.png) | ![Fit report](screenshots/05-fit-report.png) |

## Quick start

Needs Python 3.12+ with [uv](https://docs.astral.sh/uv/), and Node 22+.

```bash
cp .env.example .env            # put your key in LLM_API_KEY (any OpenAI-compatible endpoint works)

# backend (first start downloads the ~130 MB embedding model)
cd backend && uv sync && uv run python -m app.main            # http://127.0.0.1:8000

# frontend, in a second terminal (dev server proxies /api to the backend)
cd frontend && npm ci && npm run dev                          # http://127.0.0.1:5173
```

Click **Try with demo data** to load the sample resume and three postings from [`samples/`](samples).

No API key? Run the stub and point the app at it:

```bash
cd backend && uv run uvicorn tests.stub_llm:app --port 8020 &
LLM_BASE_URL=http://127.0.0.1:8020/v1 LLM_MODEL=stub uv run python -m app.main
```

Docker (the Dockerfile uses Red Hat UBI images, so run `docker login registry.redhat.io` first):

```bash
docker compose up --build       # http://127.0.0.1:8000
```

Checks:

```bash
cd backend  && uv run ruff check . && uv run mypy app && uv run pytest --cov=app    # 61 tests, 93% coverage
cd frontend && npm run typecheck && npm test && npm run build
cd backend  && uv run python -m evals.retrieval                                      # real embedding model
```

## Architecture

```mermaid
flowchart LR
  UI[React + TS UI] -- SSE / REST --> API[FastAPI]
  subgraph Backend
    API --> ING[Ingestion<br/>parse, chunk, embed]
    API --> CHAT[Chat pipeline]
    API --> FIT[Fit report]
    ING --> IDX[(Per-session index<br/>numpy cosine + BM25)]
    CHAT --> RET[Retriever<br/>per-document budget]
    RET --> IDX
    CHAT --> G[Guardrails]
    CHAT --> LLM[OpenAI-compatible LLM]
    FIT --> LLM
    ING --> EMB[fastembed, local ONNX<br/>bge-small-en-v1.5]
    RET --> EMB
  end
  API --> M[/metrics + JSON logs/]
```

A chat request goes through five steps: **input guardrail → retrieve → build prompt → stream → verify
citations**. They are plain async functions in [`rag/pipeline.py`](backend/app/rag/pipeline.py), with
no orchestration framework.

Layout: `app/ingestion` (parsing, chunking), `app/rag` (embeddings, index, retriever, prompts,
guardrails, LLM client, pipeline, fit), `app/api.py` (thin routes), `app/sessions.py`,
`app/observability.py`. Tests are in `backend/tests`, the retrieval eval is in `backend/evals`.

## RAG / LLM approach and decisions

**Why two modes.** Questions like "what am I missing for Job #2" are comparative and need
evidence from two documents. A whole-document verdict ("score me against this job") doesn't
benefit from retrieval at all, because both documents fit in context and retrieval can only drop
information. So chat uses RAG, and the structured **fit report** sends the full resume and posting.

**Chunking.** Resumes and postings are short and structured, so I split on detected headings
(known section names, markdown headings, ALL-CAPS lines) and then pack paragraphs/bullets up to
900 characters. A chunk never crosses a section boundary, and bullets are kept individually. Each
chunk keeps its document label and section, which is embedded as a header and shown as the citation.
I did not use fixed-size sliding windows, because they cut "Requirements" in half.

**Embeddings: `BAAI/bge-small-en-v1.5` via fastembed (local ONNX).** Resumes are PII, so I wanted
embeddings to run without sending the document to a third party, and with no per-token cost. Small
model, CPU-only, ~130 MB. Considered: OpenAI `text-embedding-3-small` (better quality, but sends PII
out and adds a network hop per query), `bge-base/large` (better, slower, bigger image).

**Retrieval: hybrid dense + BM25 with Reciprocal Rank Fusion.** Skill matching is dominated by
exact tokens (CISSP, Kubernetes, NDCG) that dense retrieval is weak at. I measured it on 12 hand-written
questions over the sample documents (`evals/retrieval.py`, real model, top-3):

| | hit@3 | MRR |
|---|---|---|
| dense only | 0.75 | 0.69 |
| hybrid | 0.92 | 0.75 |

This is a tiny eval and the sample set is also what I tuned on, so it justifies the direction and
should not be read as a benchmark. One case still misses ("How did I improve performance?" needs
a paraphrase that neither signal catches).

**Per-document retrieval budget.** Retrieval runs separately for the resume and each posting
(equal share, minimum 2 chunks), so one verbose posting cannot crowd out the others. If the
question names "Job #2", retrieval is restricted to the resume and that posting.

**Vector database: none, on purpose.** A session is one resume plus ≤5 postings, which is a few
dozen chunks. A numpy matrix with brute-force search is exact, has no moving parts, and is faster
than a network round trip. The index sits behind `SessionIndex`, so swapping in pgvector or Qdrant is
a localized change, and I describe when I would do it below.

**LLM.** Any OpenAI-compatible endpoint (default `gpt-4.1-mini`, configurable). I chose the
protocol over a vendor SDK so the same code runs against OpenAI, Azure OpenAI, or a local
Ollama/vLLM model for teams that cannot send resumes out. Temperature is 0.2 for chat and 0 for the
fit report. I haven't compared models on this task.

**Orchestration: no framework.** The flow is a straight line of five steps. LangChain/LlamaIndex
would add abstraction without removing any code, and would make it harder to see exactly what goes
into the prompt. The cost is that I own things like retries and streaming myself.

**Prompt and context management.**
- Prompts are in one file ([`rag/prompts.py`](backend/app/rag/prompts.py)) with a version string that
  is logged on every answer.
- Retrieved passages are numbered `[S1]…`, fenced in `<passages>`, and the system prompt says they
  are untrusted data. The model must cite and must say "not shown on your resume" rather than guess.
- Context budget: 12k characters of passages, last 6 turns of history, 1,000-character questions.
  Passages are ordered resume first, then by job and position.

**Guardrails** (deterministic, unit-tested; defence in depth, not a guarantee):
- *Input:* length limits, and a regex check for instruction-hijack phrasing, rejected before the
  LLM is called. I wrote a test for false positives after "how do I *act as* a team lead" was
  wrongly blocked by my first version.
- *Documents are untrusted:* uploads are scanned for injection phrasing and the user gets a warning;
  the text is still fenced as data. The model has no tools, so the worst case of an injection is a bad
  answer, not an action.
- *Output:* every `[S#]` the model emits is checked against the passages it was actually given.
  Invented ids are stripped and counted. Answers with no citations are flagged in the UI.
- *Fit report:* JSON is validated with pydantic (one retry on malformed output), and each "matched
  skill" must include an evidence quote that is verified verbatim against the resume. Unverified
  claims are shown with an "unverified" badge.
- *Rendering:* model-supplied links are never rendered. Only internal citation anchors become chips.

**Observability.** JSON logs to stdout (request id, route, status, latency, and per-answer
retrieval ms, time to first token, token counts, cited/invalid citation counts, model and prompt
version). No document text or questions are logged. Prometheus metrics are at `/metrics`: request
latency, retrieval latency, first-token latency, token counts, guardrail blocks, citation outcomes.

**Quality controls.** Unit tests for chunking, index, guardrails and sessions; API tests with a
deterministic embedder and a scripted LLM; the OpenAI client wrapper is tested over HTTP;
retrieval eval script. There is no automated eval of answer quality (see below).

## Key technical decisions

- **Sessions are in memory with a 2-hour TTL and an explicit delete.** Resumes are PII, so
  "don't persist" is the safest default. Cost: a restart loses everything, and it can't scale
  horizontally as is.
- **The session id is the only credential** (24 random bytes). No accounts. This is fine for a demo,
  not for production.
- **Streaming over SSE** so the first tokens appear quickly. Guardrail and "no LLM configured" errors are
  raised before the first byte, so they get real HTTP status codes instead of a mid-stream error.
- **Job numbers are never reused.** "Job #2" in an earlier answer keeps meaning the same posting even
  after you delete another one. A failed upload doesn't consume a number.
- **Single image serving UI and API.** One container, one origin, no CORS to configure.
- **Service binding.** The app binds `127.0.0.1` by default; in a container it binds the container's
  interface address rather than `0.0.0.0` (see `_bind_host`), and compose publishes only on loopback.
  I followed that as an explicit security rule; the trade-off is that the bind logic is more
  surprising than the usual one-liner.

## Productionizing (AWS / GCP / Azure / Cloudflare)

Roughly in the order I would do it:

1. **Externalize state.** Sessions to Redis (with TTL), vectors to pgvector or Qdrant (filter by
   session/user id), original files to encrypted object storage if they need to be kept at all. This
   makes the app stateless so it can run multiple replicas behind a load balancer.
2. **Separate ingestion from serving.** Embedding is CPU-bound; put it on a queue (SQS / Pub-Sub) with
   a worker pool, and let the API return "processing" quickly. Chat pods stay light.
3. **Auth and multi-tenancy.** OIDC login (the security baseline calls for SAML/OIDC), per-user data
   scoping, per-user rate limits and quotas on LLM spend. Right now anyone who can reach the app can
   consume LLM tokens.
4. **Deploy.** Containers on ECS Fargate / Cloud Run / Container Apps (or Kubernetes if there is
   already a platform), TLS terminated at the load balancer, secrets from the cloud secrets manager
   instead of `.env`, autoscaling on CPU and concurrent streams. Cloudflare could front it for WAF and
   caching of the static UI, but the Python/ONNX backend isn't a Workers fit.
5. **LLM access.** Use the cloud-native endpoint (Azure OpenAI, Bedrock, Vertex) behind the same
   protocol, with timeouts, circuit breaking and a fallback model. Add a content-safety filter.
6. **Observability.** Ship the existing JSON logs and Prometheus metrics to the platform stack
   (CloudWatch/Managed Prometheus/Grafana), add OpenTelemetry traces across retrieve and LLM
   spans, and alert on error rate, first-token latency, and citation-failure rate.
7. **Delivery and security.** CI running lint, types, tests, SAST and dependency scanning; image
   scanning and signing; infrastructure as code; PII retention policy and data-deletion audit trail.
8. **Cost.** Cache fit reports per (resume hash, job hash); use a small model by default and escalate
   only for the report.

## Engineering standards

**Followed:** typed Python (mypy strict clean) and TypeScript (strict); ruff lint and format; 61 backend
tests at 93% coverage plus frontend tests; thin routes with logic in services; dependency injection
of the embedder and LLM so tests run offline; config from environment only; no secrets in
code or git (`.env` is ignored, `.env.example` is provided); input size and type limits that check
content, not only extensions; generic error messages to clients with detail only in server logs;
security headers; non-root, read-only, capability-dropped container config; lockfiles committed
(`uv.lock`, `package-lock.json`).

**Skipped or incomplete:**
- No CI pipeline is configured.
- No dependency vulnerability scan in CI (`npm audit` was clean when I ran it; I have not run a
  Python equivalent).
- The container build is untested (see above).
- No end-to-end browser test; the UI was exercised by a scripted headless browser run only.
- No auth or rate limiting.
- No scanned-PDF/OCR support: such files are rejected with a message.
- Frontend tests cover only citation parsing and SSE parsing, not components.
- `evals/retrieval.py` isn't wired into CI.

## What I'd do differently with more time

<!-- TODO(author): write your own priorities here. Candidates from my notes: -->

- Evaluate answer quality properly: a labelled question set with an LLM-as-judge for groundedness and
  a human spot-check, run in CI to catch prompt regressions. This is the biggest gap.
- Run against a real model and tune the prompts. Everything prompt-related is untested against a real LLM.
- Reranking (cross-encoder) and query rewriting for follow-ups like "what about the second one?".
- Resume-aware parsing (skills lists, date ranges → years of experience) so the experience-alignment
  analysis is computed rather than only judged by the model.
- A real vector store and Redis, per the production plan.
- Voice or OCR input, and an export of the fit reports.

## Known limitations

- Answer quality is unverified against a real LLM (see the top of this file).
- Section detection is heuristic; resumes with unusual headings fall back to one big "Header" section.
- Injection detection is a phrase list and can be bypassed. The structural mitigations matter more.
- The fit score is the model's judgement and is not calibrated.
- English only.
