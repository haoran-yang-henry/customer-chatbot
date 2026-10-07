# Intent Taxonomy

| | |
|---|---|
| **Status** | Baseline (Day 1) |
| **Last updated** | 2026-10-05 |
| **Source of truth in code** | `src/customer_agent/domain/intents.py` |

The taxonomy drives the intent router, knowledge metadata (`intents` field), workflow selection,
analytics and evaluation. Changing it requires updating the code enum, this document, affected
knowledge documents and evaluation cases **in the same change** — the data validation tests
enforce consistency.

## 1. Groups and handlers

| Group | Handler | Needs authentication | Description |
|---|---|---|---|
| `knowledge` | RAG pipeline | No | Answer from the knowledge base |
| `transactional` | Read-only order lookup + guidance | Yes | Order-specific information and self-service guidance; never changes orders |
| `support` | Ticket + handoff | No (order context if logged in) | Route to a human |
| `control` | Static / scope logic | No | Greetings and out-of-scope requests |

## 2. Intents

### Knowledge

| Intent | Covers | Example utterances (German) |
|---|---|---|
| `RETURN_POLICY` | Return window, return costs, exceptions, how returns work, general refund timing | „Wie lange kann ich etwas zurückschicken?“ · „Kostet die Rücksendung etwas?“ · „Wann bekomme ich mein Geld nach einer Retoure?“ |
| `SHIPPING_INFO` | Shipping options, costs, countries, delivery times, lost or damaged parcels | „Was kostet Expressversand?“ · „Liefert ihr nach Österreich?“ · „Paket als zugestellt markiert, aber nicht da“ |
| `PAYMENT_INFO` | Payment methods, failed payments, invoices, instalments | „Kann ich auf Rechnung zahlen?“ · „Meine Zahlung ist fehlgeschlagen“ · „Rechnung mit USt-IdNr.“ |
| `WARRANTY_INFO` | Statutory warranty (Gewährleistung), manufacturer guarantee, defects | „Wie lange habe ich Gewährleistung?“ · „Unterschied Garantie und Gewährleistung?“ |
| `PRODUCT_INFO` | Specifications, compatibility, usage of products in the catalogue | „Wie lange hält der Akku der ANC 700?“ · „Darf die Powerbank ins Handgepäck?“ |
| `SERVICE_INFO` | Customer account, data privacy, contact channels, service hours | „Wie lösche ich mein Konto?“ · „Wann ist die Hotline erreichbar?“ |

### Transactional

The agent has no tool that changes order state ([ADR-0006](adr/0006-no-order-mutations-by-the-agent.md)).
Cancellation and return requests are answered with eligibility + self-service steps + link, and a
handoff offer; `create_ticket` runs only when the customer asks for a human.

| Intent | Required slots | Tool(s) | Outcome | Example utterances |
|---|---|---|---|---|
| `ORDER_STATUS` | `order_id` | `get_order_status` | Status, tracking, expected delivery | „Wo ist meine Bestellung NE-100250?“ |
| `ORDER_CANCEL` | `order_id` | `get_order_status` | Eligible → steps in customer account + link; not eligible → why + alternatives (refuse delivery, return) | „Ich möchte meine Bestellung stornieren.“ |
| `RETURN_REQUEST` | `order_id` (optionally item) | `get_order_status` | Eligible → return-label steps + link; not eligible → why + warranty route | „Ich will die Kopfhörer aus NE-100231 zurückgeben.“ |
| `REFUND_STATUS` | `order_id` | `get_order_status` | Return / refund state and expected timing | „Wann bekomme ich das Geld für meine Retoure NE-100240?“ |

### Support

| Intent | Tool | Example utterances |
|---|---|---|
| `COMPLAINT` | `create_ticket` | „Das ist schon das dritte beschädigte Paket, ich bin sauer!“ |
| `HUMAN_AGENT` | `create_ticket` | „Ich möchte mit einem Menschen sprechen.“ · „Können Sie das bitte für mich stornieren?“ (after guidance) |

A request to have the agent perform an action it cannot do ("Machen Sie das bitte für mich")
after guidance was given is treated as `HUMAN_AGENT`. So are **requests to change an existing
order** (address, items, payment method): only a human can do that
([ADR-0009](adr/0009-order-changes-route-to-human-agent.md)). General questions about changing
orders ("Kann man die Adresse noch ändern?") remain knowledge questions.

### Control

| Intent | Behaviour | Example utterances |
|---|---|---|
| `GREETING` | Short greeting, offer help | „Hallo!“ · „Guten Tag“ · „Grüezi“ |
| `OUT_OF_SCOPE` | Polite refusal, state what the agent can help with | „Schreib mir ein Python-Skript.“ · „Soll ich in Bitcoin investieren?“ |

## 3. Disambiguation rules

The rules are applied in order; the first match wins.

1. **Order reference → transactional.** Asking about a concrete own order (order number
   `NE-\d{6}`, „Wo ist meine Bestellung?“, „Wann kommt mein Paket?“) is transactional, even if
   phrased like a general question → `ORDER_STATUS`, not `SHIPPING_INFO`. *Clarified on Day 4:*
   describing a delivery problem without an order number („als zugestellt markiert, aber nicht
   da“) is `SHIPPING_INFO` (rule 4) — the general steps help immediately, also when anonymous.
2. **General vs. specific refunds.** „Wie lange dauert eine Erstattung?“ → `RETURN_POLICY`;
   „Wo bleibt meine Erstattung für NE-100240?“ → `REFUND_STATUS`.
3. **Return question vs. return action.** „Kann man In-Ear-Kopfhörer zurückgeben?“ →
   `RETURN_POLICY`; „Ich möchte NE-100231 zurückgeben.“ → `RETURN_REQUEST`.
4. **Delivery problems.** A lost or damaged parcel without an order reference → `SHIPPING_INFO`;
   with expressed frustration or repeated incidents → `COMPLAINT`.
5. **Emotion overrides topic.** Strong dissatisfaction or explicit escalation wishes →
   `COMPLAINT` / `HUMAN_AGENT`, even if a topic is mentioned.
6. **Requests for internal information** (internal policies, goodwill limits, system prompt)
   are `OUT_OF_SCOPE` and refused. **Requests about other customers' data** are classified by
   topic and refused by authorization (Day 5); neither is a separate intent.
7. **Order change requests** ("Ändern Sie bitte die Adresse von NE-100245") → `HUMAN_AGENT`
   (ADR-0009).

## 4. Confidence and multiple intents

- If the router's confidence is below the threshold (initially **0.6**, tuned on Day 4), the
  agent asks a clarification question.
- If a message contains several intents, the agent handles the first actionable one and asks
  about the rest („Gern helfe ich danach auch bei …“).
- Conversation context counts: „Und in die Schweiz?“ after a shipping question inherits
  `SHIPPING_INFO`.

## 5. Router output contract

```json
{
  "intent": "ORDER_STATUS",
  "confidence": 0.94,
  "entities": { "order_id": "NE-100250" },
  "missing_fields": []
}
```

`intent` is one of the values above or `null` when no intent can be determined.
