# ADR-0007: OpenAI `text-embedding-3-small` for embeddings

- **Status:** Accepted
- **Date:** 2026-10-06
- **Deciders:** Project owner
- **Related:** [ADR-0002](0002-postgresql-pgvector-for-hybrid-retrieval.md),
  [ADR-0003](0003-german-language-strategy.md), NFR-01, requirements Q2

## Context

Hybrid retrieval needs a multilingual embedding model that handles German paraphrases, compounds,
missing umlauts and dialect (LR-01, LR-02). The corpus is small (~30 documents, ~50 chunks) and
the evaluation set has 37 retrieval cases, so embedding cost is negligible. The development
machine has 7 GB of RAM available to WSL and a 4 GB GPU.

## Options considered

1. **OpenAI `text-embedding-3-small`** — hosted, multilingual, 1536 dimensions,
   $0.02 per million tokens.
2. **OpenAI `text-embedding-3-large`** — better quality, 3072 dimensions (reducible via the
   `dimensions` parameter), $0.13 per million tokens.
3. **Local open model** (`multilingual-e5-base`, `bge-m3`) — free and offline, but adds PyTorch
   (~1–2 GB) and memory pressure next to Docker; e5 requires `query:` / `passage:` prefixes.
4. **Azure OpenAI** — same models, enterprise controls; needs an Azure resource and deployment.

## Decision

We use **option 1**, `text-embedding-3-small` with 1536 dimensions, called through the official
`openai` Python SDK behind the `EmbeddingProvider` port.

- 1536 dimensions fit pgvector's HNSW limit for the `vector` type (2000 dimensions). Option 2 at
  full size would require `halfvec` or a reduced `dimensions` value.
- The model name and dimension are configuration (`CA_EMBEDDING_MODEL`,
  `CA_EMBEDDING_DIMENSIONS`). The database column dimension is fixed by a migration; the
  application refuses to start if the provider dimension differs from the column.
- Every chunk stores the `embedding_model` that produced it; changing the model triggers
  re-embedding during the next ingestion instead of mixing vectors from different models.
- Tests use a deterministic fake provider; no test calls the OpenAI API unless marked `llm`.

## Consequences

**Positive**

- No local model infrastructure; fast to set up; good German quality out of the box.
- Swapping to `-large`, Azure OpenAI or a local model is a new adapter plus, if the dimension
  changes, a migration — and the retrieval evaluation shows whether it is worth it.

**Negative / trade-offs**

- Knowledge text and customer questions are sent to an external provider. Acceptable for
  synthetic data; a real deployment needs a data processing agreement or an EU-hosted model.
- Ingestion and retrieval depend on network and provider availability (fallbacks: Day 7).
- Requires an API key, stored only in `.env` (git-ignored) or CI secrets.

**Follow-ups**

- Backlog: compare against `text-embedding-3-large` (1024 dimensions) and a local model on the
  same retrieval evaluation.
