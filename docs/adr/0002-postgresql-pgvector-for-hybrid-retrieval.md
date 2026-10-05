# ADR-0002: PostgreSQL + pgvector for hybrid retrieval

- **Status:** Accepted
- **Date:** 2026-10-05
- **Related:** FR-10, FR-13, LR-02, NFR-01, [threat model T15](../threat-model.md)

## Context

Retrieval must combine lexical matching (exact terms such as "Nachnahme", product names, prices)
with semantic matching (paraphrases, colloquial German). Results must be filtered by status,
validity window, tenant and ACL **inside** the query so that forbidden content never leaves the
store. The system also needs relational storage for sessions, orders, tickets and the audit log.
Local development should be free and offline-capable apart from model APIs.

## Options considered

1. **PostgreSQL + pgvector + built-in full-text search** — one database for everything; German
   stemming via the `german` text search configuration; SQL filters and ranking in one query.
2. **Azure AI Search** — managed hybrid search with semantic ranker and good German analyzers;
   paid, cloud-only, a second data store next to the relational database.
3. **Dedicated vector DB (Qdrant, Weaviate) + Postgres** — strong vector features, but two stores
   and lexical search of varying quality.
4. **Elasticsearch / OpenSearch** — excellent BM25 and analyzers; heavy to run locally.

## Decision

We choose **option 1**. Hybrid search runs as two ranked candidate lists in Postgres
(`ts_rank_cd` over a `german` `tsvector`, and cosine distance over a `vector` column), merged with
Reciprocal Rank Fusion in the application. Filters are SQL `WHERE` clauses shared by both
branches.

Retrieval sits behind the `SearchRepository` port, so Azure AI Search can be added later as a
second adapter and compared on the same evaluation set (backlog item).

## Consequences

**Positive**

- Single store, transactional ingestion, simple local setup (`docker compose up`).
- Filters are enforced in the database (mitigates T05, T15).
- Learning value: SQL, full-text search internals and vector indexing are visible, not hidden.

**Negative / trade-offs**

- Postgres `ts_rank_cd` is not true BM25; acceptable for this corpus size, measured on Day 2.
- The `german` snowball stemmer does not split compound words; mitigated by embeddings
  (see ADR-0003).
- No managed semantic ranker; a cross-encoder reranker is a backlog item.

**Follow-ups**

- Day 2: choose the embedding model and dimension (ADR); create HNSW index.
