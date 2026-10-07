# ADR-0005: Explicit conversation state and validated tool calls

- **Status:** Accepted (state machine: Day 4; validated tool calls: Day 5)
- **Date:** 2026-10-05
- **Related:** FR-20 – FR-24, [threat model T06, T08](../threat-model.md),
  [ADR-0006](0006-no-order-mutations-by-the-agent.md)

## Context

Transactional intents read customer data and can open tickets. Letting the LLM decide which tool
to call and with which arguments — based only on the chat history — makes behaviour
non-deterministic, hard to test, and vulnerable to prompt injection (excessive agency, IDOR via
invented order numbers).

ADR-0006 removes all order-changing tools. The remaining tools (`get_order_status`,
`create_ticket`) still need controlled invocation.

## Options considered

1. **Autonomous tool calling**: the LLM receives tool definitions and calls them directly in a loop.
2. **Explicit state + validated proposals**: the LLM extracts intent and arguments; a state
   machine decides the next step; deterministic code validates and authorizes before any call.

## Decision

We propose **option 2**.

- Session state (`active_intent`, `state`, `slots`) is persisted and is the source of truth, not
  the chat history.
- The LLM produces a structured `ToolProposal` (validated against a schema). It never invokes
  tools itself.
- Pipeline: proposal → schema validation → authorization (`customer_id` from the verified token,
  order ownership) → eligibility policy (advisory) → tool call → audit log.
- `create_ticket` is only reachable from the `HUMAN_AGENT` / `COMPLAINT` flows, or from the
  "self-service not possible" branch of a transactional flow.

## Implementation (Day 4)

- `domain/session.py`: `SessionState` (active intent, workflow state, slots) and the pure
  function `decide(state, routing, authenticated) → Decision`; every transition is unit-tested.
- `application/routing.py`: rules first (greeting; bare order number while
  `AWAITING_ORDER_ID`), otherwise one LLM call returning intent, confidence, "about a specific
  order", a standalone query and an optional clarification question. Order numbers are always
  extracted by regex, never taken from the LLM.
- `application/chat.py` + `chat_sessions` / `chat_messages` (migration 0002): the server owns
  transcript and state; the client sends `session_id` + message.

## Consequences

**Positive**

- Authorization and eligibility rules are unit-testable and independent of model behaviour.
- Prompt injection cannot read other customers' orders or open arbitrary tickets.

**Negative / trade-offs**

- More code than autonomous tool calling; new intents need a state machine definition.
- Less flexible for open-ended tasks — acceptable for a bounded customer service domain.
