# NordElektro Customer Service Agent

A German-language customer service agent for a fictional online electronics retailer. It answers
policy and product questions from a governed knowledge base, and performs order workflows (status,
cancellation, returns) behind explicit authorization, policy checks and confirmation.

Customers write German; the agent answers in German. Code and documentation are in English
(ADR-0003).

> **Status: Day 6 of 8** — layered guardrails (ADR-0011):
> injection heuristics and an LLM classifier in front of the router, personal data masked
> before any model or log sees it, untrusted documents quoted as data, answers checked for
> leaks, language and unverifiable links or codes. Measured on a red-team set (attack success
> 2.7 %, no false positives); CI scans for secrets and vulnerable dependencies. Authenticated
> customers (Keycloak, OIDC) get order-specific guidance from read-only order data and can be
> handed over to a human with a ticket. Observability and resilience follow on Day 7.

## Documentation

The project documentation lives in a local `docs/` folder that is not under version control
(`.gitignore`); code comments and this README refer to it by name. The API contract is also
available as OpenAPI at `/docs` of the running service.

| Document | Content |
|---|---|
| `docs/development-plan.md` | Eight-day plan, definition of done, cut list, roadmap |
| `docs/requirements.md` | Use cases, functional, language and non-functional requirements |
| `docs/intent-taxonomy.md` | Intents, examples, disambiguation rules |
| `docs/architecture.md` | Context, containers, layers, flows, data model |
| `docs/threat-model.md` | STRIDE + OWASP LLM Top 10, mitigations, tests, residual risks |
| `docs/evaluation-plan.md` | Test levels, dataset schema, metrics, CI gating, findings |
| `docs/api.md` | API summary |
| `docs/adr/` | Architecture decision records (ADR-0001 – ADR-0011) |

## Quick start

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), Docker, an OpenAI API key
(embeddings, ADR-0007) and a key for an OpenAI-compatible chat endpoint (ADR-0008).

```bash
cp .env.example .env   # then set CA_OPENAI_API_KEY and CA_LLM_API_KEY
make install           # dependencies + pre-commit hooks
make check             # lint, type check, unit tests (no Docker, no API key)
make test-integration  # repository tests against a throwaway Postgres (Testcontainers)
make test-llm          # same + production scenarios with the real embedding model
make up migrate        # start Postgres + pgvector and Keycloak, apply migrations
make seed              # load the mock order system (customers, orders)
make token USER=anna.schneider  # access token for a test customer (see infra/keycloak/README.md)
make ingest            # index data/knowledge (idempotent)
make eval-retrieval    # Recall@5 / MRR for fulltext, vector and hybrid search
make eval              # end-to-end evaluation of the real agent + groundedness judge
make eval-redteam      # red-team set: attack success and guardrail false-positive rates
make audit             # gitleaks, Trivy and pip-audit, as in the CI security job
make run               # API on http://localhost:8000 (docs at /docs), live logs in the terminal
```

Operations: `GET /metrics` serves Prometheus metrics (HTTP, per-step latency, LLM tokens and
time to first token, guardrail decisions). For traces, `make up-observability` starts Jaeger
(http://localhost:16686) and `CA_OTEL_ENDPOINT=http://localhost:4318` sends one trace per request
with a span per agent step. `CA_LOG_FORMAT=json` writes one JSON object per log line, with
request, session and trace ids and personal data masked. Probes: `/health/live` (process)
and `/health/ready` (database; open circuit breakers are reported but keep the instance ready,
because the agent then answers with a handoff). When the model provider fails, timeouts, one
retry and a circuit breaker bring every turn to a handoff within 10 s; chat messages are
limited per customer or client address (`CA_RATE_LIMIT_PER_MINUTE`, HTTP 429).

## Repository layout

```text
src/customer_agent/   domain · application · infrastructure · api · evaluation
data/knowledge/       German knowledge base (Markdown + YAML front matter)
data/seed/            mock customers and orders
eval/datasets/        versioned evaluation datasets (JSONL)
docs/                 requirements, architecture, threat model, ADRs (local, not versioned)
tests/                unit and data validation tests
```

## Results

| Day | Gated pass rate | Notes |
|---|---|---|
| D1 | n/a (0 cases due) | Placeholder agent: 3/71 overall; harness and dataset validated |
| D2 | n/a (0 cases due) | Retrieval only, see below |
| D6 | **69/71 (97 %)** | Adversarial 8/8 (S-007 fixed). Red team: attack success 1/37 (2.7 %; 3/37 without the LLM classifier), guardrail false positives 0/12 (core: 0/63), no harmful output. Open: A-004, G-004 (router intents, nothing disclosed) |
| D5 | **65/66 (98 %)** | Intent 65/65, tools 14/14, sources 36/36; groundedness 29/29 on knowledge answers (uncalibrated). Open: S-007 (Day 6) |
| D4 | **45/47 (96 %)** | After review fixes (clean cache): intent accuracy 62/65 (95 %); groundedness 31/31 on Day-4 answers (uncalibrated). An earlier 47/47 was measured on a partly stale cache |
| D3 | **32/32 (100 %)** | Real agent (`deepseek-v4-flash`, prompt `answer/v2`); groundedness 23/24 on Day-3 answers (judge, uncalibrated) |

**Retrieval (Day 2, 37 cases, `text-embedding-3-small`)**

| Mode | Recall@5 | MRR |
|---|---|---|
| Full-text (German FTS) | 0.865 | 0.743 |
| Vector | 0.973 | 0.944 |
| Hybrid (RRF) | **0.973** | 0.876 |

Day 5: weighted RRF (full-text weight 0.25) reaches 1.000 / 0.954 on 36 cases and 1.000 / 0.872
on the held-out set (architecture §6.4).

## Roadmap

Day 7 (observability, resilience) and Day 8 (CI evaluation gate, final report) follow; the
roadmap beyond is in `docs/development-plan.md` §6.
