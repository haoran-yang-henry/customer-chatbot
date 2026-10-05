# ADR-0001: Modular monolith with ports and adapters

- **Status:** Accepted
- **Date:** 2026-10-05
- **Related:** [architecture.md](../architecture.md), NFR-13

## Context

The system combines several concerns: HTTP API, intent routing, retrieval, LLM calls, workflows,
guardrails, ingestion and evaluation. It is built by one developer in eight days and must be easy
to test without network access. External dependencies (LLM, embeddings, search, order system)
are likely to change — e.g. pgvector may later be compared with Azure AI Search.

## Options considered

1. **Single-file / flat application** — fastest start, but untestable and unmaintainable beyond a
   prototype.
2. **Microservices** (orchestrator, retrieval, ingestion, evaluation as separate services) —
   independent scaling and deployment, but network contracts, service discovery and distributed
   debugging cost days we do not have, with no measured need.
3. **Modular monolith with ports and adapters** — one deployable unit; layers `domain`,
   `application`, `infrastructure`, `api`; external dependencies behind `Protocol` interfaces.

## Decision

We choose **option 3**.

- `domain` is pure Python (plus Pydantic for validated value objects) and performs no I/O.
- `application` defines ports as `typing.Protocol` classes and implements use cases against them.
- `infrastructure` provides adapters (Postgres, LLM clients, filesystem) and in-memory fakes.
- `main.py` is the composition root that wires adapters into use cases.
- The API, ingestion CLI and evaluation CLI are entry points into the same package.

Pydantic in the domain layer is an accepted, deliberate coupling: it gives validated, immutable
models with little code. If the domain ever needs to be framework-free, models can be converted to
dataclasses behind the same names.

## Consequences

**Positive**

- Fast unit tests with fakes; no network in the default test run.
- Swapping a provider (LLM, search) touches one adapter and the composition root.
- One deployment, one log stream, one trace — simple to operate and debug.

**Negative / trade-offs**

- All components scale together; acceptable at the target load (NFR-09).
- Boundaries are enforced by convention and review, not by process isolation.

**Follow-ups**

- Consider `import-linter` to enforce the dependency rule automatically if violations appear.
