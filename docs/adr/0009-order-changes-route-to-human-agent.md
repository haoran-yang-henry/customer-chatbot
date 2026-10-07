# ADR-0009: Order change requests are routed to `HUMAN_AGENT`

- **Status:** Accepted
- **Date:** 2026-10-07
- **Deciders:** Project owner
- **Related:** [intent taxonomy](../intent-taxonomy.md), [ADR-0006](0006-no-order-mutations-by-the-agent.md),
  evaluation case T-011

## Context

Customers ask to change existing orders: delivery address, items, payment method. The taxonomy
had no intent for this; case T-011 ("Können Sie bitte die Lieferadresse für NE-100245 ändern?")
was left unscored. Under ADR-0006 the agent never changes orders, and there is no self-service
for order changes in the customer account — only a human agent can (sometimes) do it.

## Options considered

1. **New intent `ORDER_CHANGE`** — eligibility check (address change only before "In
   Bearbeitung") and tailored guidance, like cancellations and returns. Separate analytics.
2. **Route to `HUMAN_AGENT`** — recognise the request and hand over with a ticket.

## Decision

We choose **option 2**. Requests to change an existing order are classified as `HUMAN_AGENT`.

- The agent states briefly that it cannot change orders and hands over; from Day 5 a ticket is
  created with the order id and the requested change, because self-service is not possible.
- General *questions* about changing orders ("Kann man die Lieferadresse noch ändern?") stay
  knowledge questions and are answered from `faq-order-change`.

## Consequences

**Positive**

- One intent and one state machine fewer; simplest correct behaviour for a request only humans
  can fulfil.

**Negative / trade-offs**

- Customers do not get an order-specific eligibility answer ("too late, it is already being
  processed") before the handoff; the human agent gives it.
- Change requests are not separable from other handoffs in intent analytics. Mitigation: the
  ticket carries a `reason` field (Day 5), so they can still be counted.
- If change requests turn out to be frequent, revisit with option 1 (a new ADR superseding this).
