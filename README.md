# NordElektro Customer Service Agent

A German-language customer service agent for a fictional online electronics retailer. It answers
policy and product questions from a governed knowledge base, and performs order workflows (status,
cancellation, returns) behind explicit authorization, policy checks and confirmation.

Customers write German; the agent answers in German. Code and documentation are in English
([ADR-0003](docs/adr/0003-german-language-strategy.md)).

> **Status: Day 2 of 8** — knowledge ingestion and hybrid retrieval (Postgres full-text +
> pgvector, RRF) are in place and evaluated. The answering agent starts on Day 3. See the
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
(embeddings, [ADR-0007](docs/adr/0007-openai-embeddings.md)).

```bash
cp .env.example .env   # then set CA_OPENAI_API_KEY
make install           # dependencies + pre-commit hooks
make check             # lint, type check, unit tests (no Docker, no API key)
make test-integration  # repository tests against a throwaway Postgres (Testcontainers)
make test-llm          # same + production scenarios with the real embedding model
make up migrate        # start Postgres + pgvector, apply migrations
make ingest            # index data/knowledge (idempotent)
make eval-retrieval    # Recall@5 / MRR for fulltext, vector and hybrid search
make eval              # end-to-end evaluation against the current agent
make run               # API on http://localhost:8000 (docs at /docs)
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

**Retrieval (Day 2, 37 cases, `text-embedding-3-small`)**

| Mode | Recall@5 | MRR |
|---|---|---|
| Full-text (German FTS) | 0.865 | 0.743 |
| Vector | 0.973 | 0.944 |
| Hybrid (RRF) | **0.973** | 0.876 |

## Roadmap

See [development plan §6](docs/development-plan.md#6-explicitly-out-of-scope-roadmap).
