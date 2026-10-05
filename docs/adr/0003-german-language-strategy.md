# ADR-0003: German language strategy

- **Status:** Accepted
- **Date:** 2026-10-05
- **Related:** LR-01 – LR-05, FR-01

## Context

Customers write in German — often colloquially, without umlauts, with typos or Austrian/Swiss
expressions. German compounds (`Rücksendekosten`) hurt lexical retrieval. The team (and future
readers of this repository) works in English. We need a consistent rule for which language is used
where.

## Options considered

1. **Everything in German** (code, docs, prompts, knowledge).
2. **Everything in English, translate at the edges** (translate the question to English, answer in
   English, translate back) — loses nuance, doubles LLM calls, translation errors in legal terms
   (Gewährleistung ≠ warranty ≠ Garantie).
3. **Split by audience**: customer-facing content in German, engineering artefacts in English.

## Decision

We choose **option 3**.

| Artefact | Language |
|---|---|
| Code, identifiers, logs, docs, ADRs, commit messages | English |
| Knowledge base content | German |
| Customer-facing answers | German, formal "Sie" |
| Answer-generation prompts | English instructions, explicit "respond in German" rule, German examples |
| Evaluation dataset utterances and expected phrases | German |

Retrieval for German:

- Full-text search uses the Postgres `german` configuration (stemming, German stop words).
- Queries are normalised before lexical search: lowercasing and umlaut transliteration both ways
  (`ae ↔ ä`, `oe ↔ ö`, `ue ↔ ü`, `ss ↔ ß`).
- A multilingual embedding model covers paraphrases, compounds and dialect (chosen on Day 2).

Input in another language is understood but answered in German (LR-03).

## Consequences

**Positive**

- No translation layer; legal and policy terms stay exact.
- Engineering artefacts remain accessible to non-German-speaking reviewers.

**Negative / trade-offs**

- Prompt authors must keep instructions (English) and output requirements (German) clearly separate.
- Deterministic `must_contain` checks depend on German phrasing; groups of alternatives are
  used to reduce brittleness.

**Follow-ups**

- Day 2: measure recall on the `language` category specifically.
- Day 6: add an output check that the answer language is German.
