# NordElektro Customer Service Agent

A German-language customer service agent for a fictional online electronics retailer. It answers
policy and product questions from a governed knowledge base, and performs order workflows (status,
cancellation, returns) behind explicit authorization, policy checks and confirmation.

Customers write German; the agent answers in German. Code and documentation are in English
([ADR-0003](docs/adr/0003-german-language-strategy.md)).

> **Status: Day 1 of 8** — requirements, design, synthetic data, evaluation dataset and the project
> skeleton are in place. The agent itself starts on Day 3. See the
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

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), Docker (from Day 2).

```bash
make install   # dependencies + pre-commit hooks
make check     # lint, type check, tests
make eval      # run the evaluation dataset against the current agent
make run       # API on http://localhost:8000 (docs at /docs)
make up        # start Postgres + pgvector (Day 2+)
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

## Roadmap

See [development plan §6](docs/development-plan.md#6-explicitly-out-of-scope-roadmap).
