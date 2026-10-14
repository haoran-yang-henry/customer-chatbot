# ADR-0011: Layered guardrails — deterministic checks first, an LLM classifier as second opinion

- **Status:** Accepted
- **Date:** 2026-10-12
- **Deciders:** Project owner
- **Related:** [threat model](../threat-model.md) T01, T02, T07, T09, T10, T12;
  [ADR-0005](0005-explicit-state-and-validated-actions.md), [ADR-0006](0006-no-order-mutations-by-the-agent.md);
  NFR-07, NFR-08

## Context

Until Day 5 the agent was protected mainly by its design: no tool changes an order (ADR-0006),
identity comes only from the token (ADR-0010), retrieval filters run in SQL, and the router
classifies "change your instructions" as `OUT_OF_SCOPE`. Every remaining defence against
prompt injection depended on a single LLM call behaving well — the router, or the answer model
obeying its prompt. Day 6 adds independent layers and measures them (attack success rate,
false-positive rate) on a red-team set.

Three questions had to be decided:

1. Which checks run where, and in which order?
2. How is a second opinion on prompt attacks obtained — heuristics only, a dedicated classifier
   model, or the existing LLM with a dedicated prompt?
3. What happens when the classifier fails?

## Options considered

**Prompt-attack detection**

1. **Heuristics only** (German + English patterns) — fast, free, explainable, but blind to
   paraphrases ("vergiss kurz, was man dir eingeschärft hat").
2. **Dedicated classifier model** (Llama Prompt Guard 2, ProtectAI DeBERTa) — purpose-built,
   but adds PyTorch and model hosting to a service that has no GPU and no model runtime, and
   these models are trained mostly on English.
3. **Heuristics, then the existing LLM with a dedicated guard prompt** — multilingual, no new
   infrastructure; costs one extra call per turn, and shares weaknesses with the main model.

**Classifier failure**

- **Fail closed** (refuse when no verdict) — every provider hiccup becomes a refusal.
- **Fail open** (continue without a verdict) — relies on the layers behind the classifier.

## Decision

Option 3, fail open, in this pipeline (each layer is a pure function returning a
`GuardrailDecision` except the classifier; the orchestrator decides what the customer sees):

| Layer | Where | Checks | Action |
|---|---|---|---|
| Input | `application/guardrails/input.py` | NFKC, invisible characters removed (Unicode tag "ASCII smuggling" decoded and scored), size, weighted injection signals (block at 1.0), IBAN / card (checksums) / e-mail / phone / secrets masked | block or redact |
| Classifier | `application/guardrails/classifier.py`, `prompts/guard/v1.yaml` | Attack probability ≥ 0.7 blocks; runs **concurrently** with the router; skipped for rule-routed messages | veto the step |
| Scope | router (Day 4) + `decide()` | Out of scope → refusal; low confidence → clarification; `third_party` and `money_demand` → refusal | refuse / clarify |
| Action | orchestrator (Day 5) | No order mutations exist; lookups ownership-checked in SQL; **ticket limit** (3 per customer per 24 h) | pending-ticket reply |
| Evidence | `application/guardrails/evidence.py` | Untrusted documents: instruction-like lines dropped, the rest quoted as data | sanitise |
| Output | `application/guardrails/output.py` | Citation, size, secrets, prompt leak (8-gram overlap), German, links / codes / numbers not in trusted evidence (sentence removed), payment data masked; the router's clarification question gets the same checks | block or redact |

Further rules:

- **Input personal data is masked before the router, the answer model, the transcript and the
  logs** see it (data minimisation, GDPR Art. 5(1)(c)); the customer is told when payment data
  was removed. The customer account holds the contact data a human colleague needs.
- **A heuristic block does not call any model** and is classified `OUT_OF_SCOPE` (taxonomy: an
  attempt to change the instructions is out of scope). A classifier veto keeps the router's
  intent and uses the most specific refusal (`third_party`, `money_demand`, else generic).
- **Refusals look like any out-of-scope refusal**, so an attacker learns nothing about which
  detector fired or how close the attempt came.
- **Streaming:** output checks need the complete answer, so they run on the full text; the
  `final` SSE event is authoritative (unchanged contract since Day 3).

## Consequences

**Positive**

- No single LLM call is the only defence. Measured on the Day 6 red-team set (evaluation plan
  §5.5): attack success 1/37 with all layers, 3/37 without the classifier, no harmful output in
  either configuration. The classifier is decisive for hybrid attacks (a normal question
  carrying an injection), which the router routes by topic; the output guardrail removed a
  phishing link the answer model had inserted when the classifier was off.
- The heuristics overlap with classifier and router (13 of their 14 blocks would also have been
  refused by either), but they cost nothing, add no latency and keep working when a model or
  prompt regresses.
- Detectors are deterministic and unit-tested; signal ids appear in logs and reports.
- The classifier adds cost but no latency (it runs alongside the router).

**Negative / trade-offs**

- One extra LLM call per turn that is not rule-routed (cost; NFR-14 checked on Day 7).
- The classifier shares the provider and model family with the router — correlated failures
  are possible; a different model for the `guard` role is a configuration change away.
- Fail-open: during a classifier outage only heuristics, router and output checks remain.
- Pattern lists need maintenance; German compounds and word boundaries are a recurring source
  of misses and false positives (documented next to each pattern).
- Answers that are redacted after streaming were briefly visible to clients that render deltas
  before the `final` event.
