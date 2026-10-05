# ADR-0006: No order mutations by the agent — guidance and human handoff

- **Status:** Accepted
- **Date:** 2026-10-05
- **Deciders:** Project owner
- **Related:** FR-20 – FR-24, UC-08, UC-09, [threat model T07, T08](../threat-model.md),
  amends [ADR-0005](0005-explicit-state-and-validated-actions.md)

## Context

The initial design let the agent cancel orders and create returns through a validated action
pipeline (proposal → authorization → policy → confirmation → execution). Order state changes
have direct financial and legal impact. Any defect in intent detection, argument extraction,
authorization or confirmation handling — or a successful prompt injection — would change real
orders. The shop already offers self-service for cancellations and returns in the customer
account, and human agents handle exceptions.

## Options considered

1. **Agent executes changes** behind validation, policy and explicit confirmation.
2. **Agent executes low-risk changes only** (e.g. cancel before shipping), hands over the rest.
3. **Agent never changes order state.** It reads order data, evaluates eligibility, explains the
   self-service steps with links and policy references, and hands over to a human on request.

## Decision

We choose **option 3**.

- The agent's tool set is: `get_order_status` (read-only, ownership-checked) and `create_ticket`.
  There is no `cancel_order`, `create_return`, refund or purchase tool.
- For cancellation and return requests the agent answers in three steps:
  1. eligibility for the specific order (deterministic policy code, advisory only),
  2. self-service instructions with a link to the customer account and the policy document,
  3. an offer to connect the customer with a human.
- A ticket is created only when the customer asks for a human, or when self-service is not
  possible (e.g. address change after payment). The ticket carries a German summary, intent,
  order id and eligibility result so the human agent does not start from zero.

## Consequences

**Positive**

- Removes the capability behind T07 (unauthorized or manipulated actions) and most of T08
  (excessive agency): there is nothing to trick the agent into executing.
- Far less safety-critical code: no confirmation state, no idempotency, no rollback.
- Matches how many retailers deploy first-generation assistants; capabilities can be added later
  one by one, each with its own ADR.

**Negative / trade-offs**

- Lower automation rate: some customers will need a human for something the account page
  already offers. Mitigated by precise links and step-by-step guidance.
- Less hands-on practice with transactional tool pipelines on Day 5; Day 5 focuses instead on
  authorization for reads, the eligibility policy engine and a high-quality handoff.

**Follow-ups**

- Track "handoff after guidance" rate (Day 7) to decide whether any change should ever be
  automated.
