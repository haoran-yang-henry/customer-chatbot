# Architecture Decision Records

An ADR records one significant decision: the context, the options considered, the decision and
its consequences. Format: a lightweight variant of [MADR](https://adr.github.io/madr/)
(see [template](0000-template.md)).

**Rules**

- One decision per file, numbered sequentially: `NNNN-short-title.md`.
- Statuses: `Proposed` → `Accepted` → (`Deprecated` | `Superseded by ADR-NNNN`).
- Accepted ADRs are not rewritten. To change a decision, write a new ADR that supersedes it.
- Write the ADR in the same change as the code that implements the decision.

| ADR | Title | Status | Date |
|---|---|---|---|
| [0001](0001-modular-monolith-with-ports-and-adapters.md) | Modular monolith with ports and adapters | Accepted | 2026-10-05 |
| [0002](0002-postgresql-pgvector-for-hybrid-retrieval.md) | PostgreSQL + pgvector for hybrid retrieval | Accepted | 2026-10-05 |
| [0003](0003-german-language-strategy.md) | German language strategy | Accepted | 2026-10-05 |
| [0004](0004-evaluation-first-with-versioned-datasets.md) | Evaluation first with versioned datasets | Accepted | 2026-10-05 |
| [0005](0005-explicit-state-and-validated-actions.md) | Explicit conversation state and validated tool calls | Proposed | 2026-10-05 |
| [0006](0006-no-order-mutations-by-the-agent.md) | No order mutations by the agent — guidance and human handoff | Accepted | 2026-10-05 |
