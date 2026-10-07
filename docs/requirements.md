# Requirements

| | |
|---|---|
| **Status** | Baseline (Day 1) |
| **Last updated** | 2026-10-05 |
| **Related** | [Intent taxonomy](intent-taxonomy.md) · [Architecture](architecture.md) · [Evaluation plan](evaluation-plan.md) |

Requirement IDs (`UC-`, `FR-`, `NFR-`, `LR-`) are stable and referenced from tests, evaluation
cases and the threat model. Do not renumber; deprecate instead.

## 1. Product context

**NordElektro GmbH** is a fictional German online retailer for consumer electronics
(headphones, laptops, smartphones, wearables, accessories). It ships to Germany, Austria and
Switzerland. Customers contact support in German.

The customer service agent is a chat assistant on the shop website and in the customer account.
It answers questions from an approved knowledge base and gives authenticated customers
order-specific guidance (status, how to cancel or return). It **never changes order state itself**:
cancellations, returns, refunds and purchases are done by the customer in the customer account or
by a human agent ([ADR-0006](adr/0006-no-order-mutations-by-the-agent.md)). Anything it cannot resolve is handed over to a human agent with a
ticket.

All data in this project is synthetic. Domains use `.example` (RFC 2606).

## 2. Stakeholders and actors

| Actor | Description | Goals |
|---|---|---|
| Anonymous customer | Website visitor, not logged in | Get policy and product information |
| Authenticated customer | Logged into the customer account | Additionally: order status and order-specific guidance |
| Support agent | Human employee | Receive complete, well-structured handoffs |
| Knowledge editor | Maintains policies and FAQ | Publish, version and retire knowledge safely |
| Operator / developer | Runs the system | Observe, debug and evaluate behaviour |

## 3. Use cases

| ID | Use case | Intent(s) | Type | Actor | Day |
|---|---|---|---|---|---|
| UC-01 | Ask about return policy and refund timing | `RETURN_POLICY` | Knowledge | Any | D3 |
| UC-02 | Ask about shipping options, costs, countries, delivery problems | `SHIPPING_INFO` | Knowledge | Any | D3 |
| UC-03 | Ask about payment methods, failed payments, invoices | `PAYMENT_INFO` | Knowledge | Any | D3 |
| UC-04 | Ask about statutory warranty and manufacturer guarantee | `WARRANTY_INFO` | Knowledge | Any | D3 |
| UC-05 | Ask about product specifications | `PRODUCT_INFO` | Knowledge | Any | D3 |
| UC-06 | Ask about account, privacy, contact channels | `SERVICE_INFO` | Knowledge | Any | D3 |
| UC-07 | Check the status of an own order | `ORDER_STATUS` | Transactional | Authenticated | D5 |
| UC-08 | Get guidance on cancelling an own order (eligibility, steps, link); handoff on request | `ORDER_CANCEL` | Transactional | Authenticated | D5 |
| UC-09 | Get guidance on returning an own order (eligibility, steps, link); handoff on request | `RETURN_REQUEST` | Transactional | Authenticated | D5 |
| UC-10 | Check refund status of an own return | `REFUND_STATUS` | Transactional | Authenticated | D5 |
| UC-11 | File a complaint | `COMPLAINT` | Support | Any | D5 |
| UC-12 | Request a human agent | `HUMAN_AGENT` | Support | Any | D5 |

### Out of scope

The agent politely declines and, where useful, redirects:

- Open-domain chat, jokes, creative writing.
- Programming, legal, medical, financial or investment advice.
- Politics, religion and other non-shop topics.
- Information about other customers, employees or internal processes.
- **Any change of order state by the agent**: placing orders, cancelling, creating returns,
  issuing refunds, changing address or payment method. The agent explains the self-service steps
  and hands over to a human on request (ADR-0006).
- Languages other than German as response language (see LR-03).

## 4. Functional requirements

### Conversation

- **FR-01** The agent responds in German, using the formal address "Sie".
- **FR-02** The agent supports multi-turn conversations and resolves references to earlier turns
  ("Und in die Schweiz?").
- **FR-03** When a request is ambiguous or required information is missing, the agent asks one
  targeted clarification question instead of guessing.
- **FR-04** The agent can switch topic mid-conversation without losing an unfinished workflow
  silently (it either completes or explicitly abandons it).
- **FR-05** The agent discloses that the customer is talking to an AI assistant (EU AI Act,
  transparency obligation); implemented in the greeting (Day 4).

### Knowledge answers

- **FR-10** Knowledge answers are based only on retrieved, approved, currently valid,
  customer-visible documents.
- **FR-11** Every knowledge answer cites the ids of the documents it is based on.
- **FR-12** When the knowledge base contains no answer, the agent says so and offers a handoff;
  it never invents policies, prices or specifications.
- **FR-13** When documents conflict, the newest valid approved version wins; expired, archived and
  draft documents are never used.

### Transactions

- **FR-20** Transactional intents require an authenticated customer. Anonymous users are asked to
  log in; general guidance (policy, steps) may still be given without order details.
- **FR-21** A customer can only read their own orders. The agent has **read-only** access to
  order data; it has no tool that changes order state.
- **FR-22** For cancellation and return requests, the agent (1) checks eligibility for the
  specific order, (2) explains the self-service steps with a link to the customer account and the
  relevant policy, and (3) offers a human handoff. A ticket is created only when the customer asks
  for a human or self-service is not possible.
- **FR-23** Eligibility rules (cancellable status, 30-day return window, return exceptions) are
  evaluated by deterministic code, not by the LLM. The result is advisory; the binding decision is
  made by the order system or a human agent.
- **FR-24** Every order lookup and every ticket creation is audit-logged with customer, tool,
  arguments, result and timestamp.

### Handoff

- **FR-30** Complaints and explicit requests for a human create a ticket containing a German
  summary, detected intent, relevant order ids and the conversation transcript.
- **FR-31** Outside service hours the agent states when a human will respond.

### Operations

- **FR-40** Every request carries a `request_id`, `session_id` and `trace_id` that appear in logs
  and traces.
- **FR-41** Knowledge can be re-ingested idempotently.

## 5. Language requirements

- **LR-01** Understand standard German including colloquial phrasing, missing umlauts
  (`zurueck`, `Oesterreich`), lowercase input, common typos, and Austrian/Swiss regional
  expressions ("Packerl", "Grüezi").
- **LR-02** Handle German compound nouns in retrieval (`Rücksendekosten` ↔ `Kosten der
  Rücksendung`).
- **LR-03** Respond in German regardless of input language. If the customer writes in another
  language, answer in German and mention once that support is offered in German.
- **LR-04** Use German number, currency and date formats (`49,99 €`, `6. Oktober 2026`).
- **LR-05** Codebase, documentation and logs are in English; customer-facing text and knowledge
  content are in German.

## 6. Non-functional requirements

Targets are chosen to be **measurable within this project**. Load-related targets measure the
system's own overhead with a mocked LLM, because LLM provider latency is outside our control.

| ID | Quality | Target | Measured by | Day |
|---|---|---|---|---|
| NFR-01 | Retrieval quality | Recall@5 ≥ 0.85, MRR ≥ 0.70 | Retrieval eval | D2 |
| NFR-02 | Answer quality | ≥ 85 % of applicable eval cases pass | Answer eval | D3 |
| NFR-03 | Groundedness | ≥ 95 % of knowledge answers judged grounded | LLM judge (calibrated) | D3 |
| NFR-04 | Intent accuracy | ≥ 90 % on cases with an expected intent | Intent eval | D4 |
| NFR-05 | Authorization | 0 cross-customer data disclosures or actions | Eval + unit tests | D5 |
| NFR-06 | Policy correctness | 100 % of policy rules covered by deterministic tests | Coverage report | D5 |
| NFR-07 | Safety | Attack success rate ≤ 5 % on red-team set | Red-team eval | D6 |
| NFR-08 | Over-blocking | Guardrail false-positive rate ≤ 3 % on normal cases | Eval | D6 |
| NFR-09 | Latency (system overhead) | P95 ≤ 300 ms excluding LLM time, at 20 concurrent users | k6 with mocked LLM | D7 |
| NFR-10 | Latency (end to end) | Time to first token P95 ≤ 3 s | Traces | D7 |
| NFR-11 | Resilience | LLM outage → handoff message within 10 s, no 5xx | Fault-injection tests | D7 |
| NFR-12 | Privacy | No raw PII (email, phone, IBAN) in logs | Log redaction tests | D7 |
| NFR-13 | Maintainability | ruff + mypy strict clean; CI < 5 min | CI | D1 |
| NFR-14 | Cost | ≤ €0.01 average LLM cost per conversation turn | Token metrics | D7 |

## 7. Constraints and assumptions

- Solo developer, eight working days (see [development plan](development-plan.md)).
- Single tenant (`nordelektro`) in v1; tenant id is still carried in data and filters so that
  multi-tenancy is a configuration, not a redesign.
- Order system and ticket system are mocked inside this repository.
- Evaluation reference date is **2026-10-05**; date-dependent rules use an injectable clock.
- LLM and embedding providers are external APIs behind ports; provider choice is an ADR on Day 2/3.

## 8. Open questions

| # | Question | Decide by |
|---|---|---|
| ~~Q1~~ | ~~Which LLM provider?~~ Decided: OpenAI-compatible endpoint, OpenCode Go `deepseek-v4-flash` for development ([ADR-0008](adr/0008-llm-provider-opencode-go.md)) | D3 |
| ~~Q2~~ | ~~Which embedding model?~~ Decided: OpenAI `text-embedding-3-small` ([ADR-0007](adr/0007-openai-embeddings.md)) | D2 |
| Q3 | Is a web UI needed, or is the API + Swagger sufficient for the demo? | D3 |
