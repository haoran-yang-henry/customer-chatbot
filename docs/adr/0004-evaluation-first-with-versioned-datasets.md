# ADR-0004: Evaluation first with versioned datasets

- **Status:** Accepted
- **Date:** 2026-10-05
- **Related:** [evaluation-plan.md](../evaluation-plan.md)

## Context

LLM system behaviour changes when code, prompts, models, chunking, embeddings or knowledge change.
Unit tests alone cannot detect most of these regressions. Without a fixed reference, quality
claims are anecdotal.

## Options considered

1. **Manual testing in the chat UI** — quick, but not repeatable and not comparable over time.
2. **Hosted evaluation platform** (Langfuse datasets, Braintrust, etc.) as the source of truth —
   rich UI, but external dependency and data outside the repository.
3. **Datasets as JSONL in the repository + own harness**, optionally exporting results to a
   platform later.

## Decision

We choose **option 3**, and we create the dataset on Day 1, before the features exist.

- Datasets live in `eval/datasets/*.jsonl`, reviewed like code.
- Cases are validated by a Pydantic schema and by data tests (referenced documents and customers
  must exist).
- Each case has a `target_day` so that CI can gate only on capabilities that should already work.
- Deterministic checks (`must_contain` groups, `must_not_contain`, behaviour, sources, tool) come
  first; LLM-as-judge only where necessary and only after calibration.
- Results are stored as JSON with configuration metadata to make runs comparable.

## Consequences

**Positive**

- Every change can be compared against the same baseline; the D1→D8 curve becomes a project
  artefact.
- Threats and requirements are traceable to concrete cases.

**Negative / trade-offs**

- A ~70-case set is small; metrics have wide confidence intervals. Grow the set as bugs are found.
- Keyword checks can be brittle with free-form German text; mitigated by alternative groups.

**Follow-ups**

- Day 8: tag a smoke subset for pull requests.
