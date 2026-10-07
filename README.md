# NordElektro Customer Service Agent

A German-language customer service agent for a fictional online electronics retailer. It answers
policy and product questions from a governed knowledge base, and performs order workflows (status,
cancellation, returns) behind explicit authorization, policy checks and confirmation.

Customers write German; the agent answers in German. Code and documentation are in English
([ADR-0003](docs/adr/0003-german-language-strategy.md)).

> **Status: Day 4 of 8** — the agent routes each message to an intent (rules first, then an
> LLM), keeps explicit conversation state in Postgres, resolves follow-up questions, asks for
> missing order numbers, and answers German questions from the governed knowledge base with
> citations ([API](docs/api.md)). Order lookup and human handoff follow on Day 5. See the
> [development plan](docs/development-plan.md).

## Documentation

| Document | Content |
|---|---|
| [Development plan](docs/development-plan.md) | Eight-day plan, definition of done, cut list |
| [Requirements](docs/requirements.md) | Use cases, functional, language and non-functional requirements |
| [Intent taxonomy](docs/intent-taxonomy.md) | Intents, examples, disambiguation rules |
| [Architecture](docs/architecture.md) | Context, containers, layers, flows, data model |
| [Threat model](docs/threat-model.md) | STRIDE + OWASP LLM Top 10, mitigations, tests |
| [Evaluation plan](docs/evaluation-plan.md) | Test levels, dataset schema, metrics, CI gating |
| [ADRs](docs/adr/README.md) | Architecture decision records |

## Quick start

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), Docker, an OpenAI API key
(embeddings, [ADR-0007](docs/adr/0007-openai-embeddings.md)) and a key for an OpenAI-compatible
chat endpoint ([ADR-0008](docs/adr/0008-llm-provider-opencode-go.md)).

```bash
cp .env.example .env   # then set CA_OPENAI_API_KEY and CA_LLM_API_KEY
make install           # dependencies + pre-commit hooks
make check             # lint, type check, unit tests (no Docker, no API key)
make test-integration  # repository tests against a throwaway Postgres (Testcontainers)
make test-llm          # same + production scenarios with the real embedding model
make up migrate        # start Postgres + pgvector, apply migrations
make ingest            # index data/knowledge (idempotent)
make eval-retrieval    # Recall@5 / MRR for fulltext, vector and hybrid search
make eval              # end-to-end evaluation of the real agent + groundedness judge
make run               # API on http://localhost:8000 (docs at /docs), live logs in the terminal
```

## Repository layout

```text
src/customer_agent/   domain · application · infrastructure · api · evaluation
data/knowledge/       German knowledge base (Markdown + YAML front matter)
data/seed/            mock customers and orders
eval/datasets/        versioned evaluation datasets (JSONL)
docs/                 requirements, architecture, threat model, ADRs
tests/                unit and data validation tests
```

## Results

| Day | Gated pass rate | Notes |
|---|---|---|
| D1 | n/a (0 cases due) | Placeholder agent: 3/71 overall; harness and dataset validated |
| D2 | n/a (0 cases due) | Retrieval only, see below |
| D4 | **47/47 (100 %)** | Intent accuracy 62/65 (95 %); groundedness 27/28 on Day-4 answers (uncalibrated); router `v2`, answer `v2`, reasoning off |
| D3 | **32/32 (100 %)** | Real agent (`deepseek-v4-flash`, prompt `answer/v2`); groundedness 23/24 on Day-3 answers (judge, uncalibrated) |

**Retrieval (Day 2, 37 cases, `text-embedding-3-small`)**

| Mode | Recall@5 | MRR |
|---|---|---|
| Full-text (German FTS) | 0.865 | 0.743 |
| Vector | 0.973 | 0.944 |
| Hybrid (RRF) | **0.973** | 0.876 |

## Roadmap

See [development plan §6](docs/development-plan.md#6-explicitly-out-of-scope-roadmap).
