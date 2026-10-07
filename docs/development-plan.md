# Development Plan

| | |
|---|---|
| **Status** | Active |
| **Owner** | Project author |
| **Started** | 2026-10-05 |
| **Planned duration** | 8 working days |
| **Last updated** | 2026-10-05 |

## 1. Purpose

This document defines how the German-language customer service agent is built in eight days:
what is delivered each day, how "done" is judged, and what is deliberately left out.

The project has two goals of equal weight:

1. **Product goal** — a production-shaped customer service agent that understands German customer
   requests and resolves them in German: answering policy and product questions from a governed
   knowledge base, and executing order workflows (status, cancellation, returns) safely.
2. **Learning goal** — practise professional software engineering around an LLM system:
   documentation standards (requirements, ADRs, threat model, evaluation plan), a layered codebase,
   automated quality gates, and fluency in the core frameworks (FastAPI, Pydantic, PostgreSQL /
   pgvector, pytest, GitHub Actions, OpenTelemetry).

"Production-grade" in an eight-day solo project means **production-shaped**: every capability that
exists is tested, observable, secured and documented, even if the scale is small. Breadth is cut
before quality is.

## 2. Guiding principles

1. **Vertical slices.** Every day ends with a runnable system, green CI and an updated evaluation
   result. No day leaves the system broken.
2. **Evaluation first.** The evaluation dataset exists before the features it measures. Every change
   to code, prompts, chunking, models or knowledge is judged against it.
3. **Explicit over implicit.** Conversation state, workflow steps and authorization are code, not
   LLM behaviour. The LLM proposes; deterministic code decides.
4. **Security ships with capability.** A tool that changes data ships together with its
   authorization and policy checks — never "hardened later".
5. **Decisions are written down.** Every non-obvious architectural decision gets an ADR
   (`docs/adr/`). Docs are updated in the same change as the code they describe.
6. **Modular monolith first.** One deployable service with clean internal boundaries
   (ports and adapters). Split into services only when a measured need appears.

## 3. Timeline overview

| Day | Theme | Key outcome | Primary metric introduced |
|---|---|---|---|
| D1 | Problem definition, design, data, skeleton | Docs baseline, synthetic German data, eval set, CI skeleton | Eval harness runs (baseline 0 %) |
| D2 | Ingestion and hybrid retrieval | Knowledge indexed in Postgres; BM25 + vector + RRF | Recall@5, MRR |
| D3 | Grounded answering and chat API | `/chat` with streaming, citations and refusal | Answer pass rate, groundedness |
| D4 | Intent routing and conversation state | Structured intent output, explicit state machine, slot filling | Intent accuracy, confusion matrix |
| D5 | Order guidance and handoff | Read-only order lookup, eligibility policy, self-service guidance, tickets | Policy test coverage, IDOR refusals |
| D6 | Guardrails and red teaming | Input / scope / retrieval / action / output guardrails | Attack success rate, false-positive rate |
| D7 | Observability and resilience | Traces, redacted structured logs, timeouts, fallbacks, load test | P95 overhead, per-step latency |
| D8 | CI evaluation gate, documentation, demo | PRs blocked on eval regressions; final report | Day-over-day quality curve |

Each day's results are stored in `eval/results/` so the final README can show quality over time.

## 4. Daily plan

Each day lists: goal, scope, deliverables, definition of done (DoD) and learning focus.

### Day 1 — Problem definition, design, data, skeleton

**Goal.** Know exactly what is being built, how it will be judged, and have an empty but
production-shaped repository that every later day plugs into.

**Scope.**
- Requirements: use cases, scope boundaries, measurable non-functional requirements, language
  requirements (German in, German out).
- Intent taxonomy with German examples and disambiguation rules.
- Architecture: context, containers, layers, request flows, data model.
- Threat model: STRIDE plus OWASP Top 10 for LLM Applications, each threat mapped to a mitigation,
  a delivery day and test cases.
- Evaluation plan and a first German evaluation dataset (~70 cases).
- Synthetic data for a fictional German online electronics retailer (*NordElektro GmbH*):
  ~30 knowledge documents including deliberately expired, draft, internal and poisoned documents,
  plus customers and orders.
- Repository skeleton: `uv` project, layered `src/` package, FastAPI health endpoint, ruff, mypy
  (strict), pytest, pre-commit, Makefile, Docker Compose (Postgres + pgvector), GitHub Actions CI.
- Evaluation harness that runs the dataset against a placeholder agent.

**Deliverables.** `docs/requirements.md`, `docs/intent-taxonomy.md`, `docs/architecture.md`,
`docs/threat-model.md`, `docs/evaluation-plan.md`, `docs/adr/0001…0005`, `data/knowledge/*.md`,
`data/seed/*.json`, `eval/datasets/core.jsonl`, application skeleton, CI workflow.

**DoD.**
- `make check` (lint, format check, type check, tests) passes.
- Data validation tests pass: every knowledge document has valid metadata, every evaluation case
  references existing sources and customers.
- `make eval` runs end to end and writes a baseline result.

**Learning focus.** Requirements engineering, ADR format, threat modelling, `uv`, Pydantic v2
models and settings, pytest structure, ruff/mypy configuration, GitHub Actions basics.

### Day 2 — Ingestion and hybrid retrieval

**Goal.** The right documents are found for German questions — measured independently of the LLM.

**Scope.**
- Ingestion pipeline as separate, testable steps:
  `load → parse → normalise → chunk → enrich metadata → validate → embed → index`.
- Postgres schema (migrations with Alembic): `knowledge_documents`, `knowledge_chunks`
  (with `tsvector` using the `german` text search configuration and a `vector` column).
- Multilingual embedding model behind an `EmbeddingProvider` port.
- Hybrid search: full-text (BM25-like `ts_rank_cd`) + vector similarity, fused with Reciprocal Rank
  Fusion. Mandatory filters: `status = approved`, validity window, tenant, ACL.
- Retrieval evaluation on the dataset's `expected_sources`.

**DoD.** Idempotent ingestion (`make ingest` twice yields the same index). Expired, draft and
internal documents are never returned to customer queries (tested). Recall@5 and MRR recorded.

**Learning focus.** SQLAlchemy 2.0 / psycopg 3, Alembic, pgvector, Postgres full-text search for
German, integration tests against a real database (Testcontainers or Compose).

### Day 3 — Grounded answering and chat API

**Goal.** First usable assistant: German answers grounded in retrieved knowledge, with citations,
and an honest "I don't know" when the knowledge base has no answer.

**Scope.**
- LLM provider behind an `LLMProvider` port; provider and model chosen in an ADR.
- Versioned prompts in `prompts/` (YAML), loaded by name and version.
- `POST /v1/chat` (request/response) and `POST /v1/chat/stream` (Server-Sent Events).
- Answer composition: German, formal "Sie", cites document ids, refuses outside the evidence.
- Minimal web UI (single page) — optional, cut first if time is short.
- Answer evaluation: `must_contain` / `must_not_contain`, behaviour checks, LLM-as-judge for
  groundedness (judge agreement checked manually on 20 samples).

**DoD.** All `target_day ≤ 3` cases evaluated; pass rate recorded; API contract documented
(OpenAPI generated, `docs/api.md` summarises it).

**Learning focus.** FastAPI dependency injection, async I/O, SSE streaming, prompt management,
LLM evaluation methodology.

### Day 4 — Intent routing and conversation state

**Goal.** Move from "RAG chatbot" to "agent": the system knows what the customer wants, what it
still needs, and where the conversation stands.

**Scope.**
- Intent router: deterministic rules first (e.g. order-number regex), then LLM structured output
  returning `intent`, `confidence`, `entities`, `missing_fields`.
- Confidence threshold → clarification question instead of a guess.
- Session state persisted in Postgres: active intent, workflow state, slots, turn history.
- Explicit state machine per transactional intent (e.g. `AWAITING_ORDER_ID`).

**DoD.** Intent accuracy and confusion matrix recorded; every state transition covered by unit
tests; multi-turn evaluation cases pass at the target rate.

**Learning focus.** Structured outputs / JSON schema, state machine design, repository pattern.

### Day 5 — Order guidance and human handoff

**Goal.** Authenticated customers get precise, order-specific help — without the agent ever
changing an order ([ADR-0006](adr/0006-no-order-mutations-by-the-agent.md)).

**Scope.**
- Mock order service (Postgres-backed, seeded from `data/seed/`), **read-only** for the agent.
- Tools: `get_order_status` (ownership-checked) and `create_ticket`.
- Eligibility policy engine: `can_cancel`, `can_return` (30-day window, exceptions) with an
  injected `Clock`; results are advisory and tailor the guidance.
- Guidance responses: eligibility → self-service steps with link to the customer account and
  policy citation → handoff offer.
- Handoff: ticket only on request (or when self-service is impossible), with German summary,
  intent, order id, eligibility result and transcript.
- Customer identity: signed token (JWT) carrying `customer_id` and `tenant_id`.
- Tool pipeline: proposal → schema validation → authorization → call → audit log (ADR-0005).

**DoD.** Eligibility and authorization rules covered 100 % by deterministic unit tests;
cross-customer access attempts in the dataset are refused without revealing that the order
exists; every order lookup and ticket is audit-logged.

**Learning focus.** Policy engine design, authentication with JWT in FastAPI, authorization
(IDOR prevention), audit logging, designing a useful human handoff.

### Day 6 — Guardrails and red teaming

**Goal.** Systematic defence across five layers, with measured effectiveness and measured cost.

**Scope.**
- Input: prompt-injection detection (heuristics + classifier), PII detection, length limits.
- Scope: in scope / out of scope / uncertain.
- Retrieval: trust-level handling (untrusted content quoted as data, never as instructions).
- Action: already enforced on Day 5; extended with rate limits on state-changing tools.
- Output: PII and secret redaction, citation presence, German-language check.
- Red-team dataset (`eval/datasets/redteam.jsonl`, ~30 cases) including indirect injection via
  the poisoned knowledge document.

**DoD.** Attack success rate and guardrail false-positive rate (on normal cases) recorded;
threat model updated with residual risks.

**Learning focus.** OWASP LLM Top 10 in practice, defence in depth, precision/recall trade-offs
for safety classifiers.

### Day 7 — Observability and resilience

**Goal.** Every request can be explained after the fact, and dependency failures degrade
gracefully.

**Scope.**
- OpenTelemetry tracing: one trace per request with spans for routing, retrieval, LLM, guardrails,
  tools. LLM-specific tracing (Langfuse or Phoenix) if time permits.
- Structured JSON logs with `trace_id`, `session_id`, `request_id`; PII redaction.
- Prometheus metrics: latency histograms, token usage, guardrail decisions, errors.
- Resilience: timeouts, retries with exponential backoff, fallback (LLM unavailable → ticket +
  handoff message), simple rate limiting.
- Exact-match FAQ cache (semantic cache stays on the roadmap).
- Load test with k6 against a mocked LLM to measure the system's own overhead.

**DoD.** Per-step latency breakdown available; P95/P99 of system overhead recorded; fallback paths
covered by tests that inject failures.

**Learning focus.** OpenTelemetry SDK, Prometheus, resilience patterns, k6.

### Day 8 — CI evaluation gate, documentation, demo

**Goal.** Quality is enforced automatically, and the project explains itself.

**Scope.**
- CI: smoke evaluation (~20 cases) on every pull request with a pass-rate threshold; full
  evaluation + red team on manual dispatch / nightly.
- Final evaluation report: D1→D8 metrics, known limitations, residual risks.
- README: architecture diagram, how to run, results table, ADR index, roadmap.
- Runbook (`docs/runbooks/`) for common operational failures.
- Three-minute demo recording.

**DoD.** A pull request that lowers the smoke-eval pass rate below threshold fails CI; a new
engineer can run the system from the README alone.

**Learning focus.** CI/CD design, technical writing for different audiences.

## 5. Definition of done (every day)

- [ ] `make check` passes locally and in CI.
- [ ] `make eval` run; result committed to `eval/results/dayN.json`.
- [ ] New behaviour covered by tests at the right level (unit / integration / eval).
- [ ] Affected docs updated (requirements, architecture, threat model, ADRs).
- [ ] Work merged to `main` through small, focused commits (trunk-based development).

## 6. Explicitly out of scope (roadmap)

Kubernetes, Terraform, message queues (Kafka), multi-agent orchestration, fine-tuning, semantic
cache, canary deployments, model routing, automated knowledge feedback loop, multi-tenant
production deployment. Each is listed in the README roadmap with a short design note.

## 7. Adjusting the plan

**If ahead of schedule**, pull from this backlog in order:

1. Cross-encoder reranker on top of hybrid retrieval (and measure the gain).
2. Azure AI Search adapter behind the same search port (compare against pgvector).
3. LLM tracing with Langfuse.
4. Prompt A/B comparison tooling in the eval harness.
5. Semantic cache for knowledge intents.

**If behind schedule**, cut in this order: D7 load test → D7 cache → D3 web UI → D6 classifier
(keep heuristics). **Never cut**: the evaluation dataset, D5 authorization checks,
the D8 CI gate.

## 8. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| LLM API cost during evaluation | Budget overrun | Small smoke set in CI; cache LLM responses for unchanged inputs; budget cap |
| LLM-as-judge noise | Misleading metrics | Prefer deterministic checks; calibrate judge on a hand-labelled sample |
| German retrieval quality (compounds, umlauts) | Low recall | `german` FTS config + multilingual embeddings; measured on Day 2 |
| Docker unavailable in WSL | No local database | Enable Docker Desktop WSL integration before Day 2 |
| Scope creep | Unfinished days | Cut list above; the daily DoD is the gate |

## 9. Change log

| Date | Change |
|---|---|
| 2026-10-05 | Initial plan |
| 2026-10-05 | D5 changed: no order mutations by the agent; guidance + human handoff (ADR-0006) |
| 2026-10-06 | D2 done. Added real-model production tests (`make test-llm`); cosine similarity kept through RRF as relevance signal |
| 2026-10-06 | D3 done. Logging foundation moved from D7 to D3; relevance gate (two-tier) from D2 measurements; LLM via OpenAI-compatible endpoint (ADR-0008); web UI cut (Swagger + curl suffice) |
| 2026-10-06 | D4 additions: conversation-aware query rewriting (replaces joined user turns); decide `ORDER_CHANGE` vs `HUMAN_AGENT` (T-011); split API/orchestrator responsibilities further |
| 2026-10-07 | D4 done. T-011 → `HUMAN_AGENT` (ADR-0009); server-side sessions (client sends `session_id` + message); AI disclosure in greeting (FR-05); `CA_LLM_REASONING_EFFORT=none` after empty-output findings |
| 2026-10-06 | Backlog additions: weighted RRF / reranker after adding new retrieval cases ("Retoure" case); human labelling of `eval/calibration/groundedness-day3.jsonl` |
