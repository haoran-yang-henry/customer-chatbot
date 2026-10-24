"""Retrieval: hybrid search with Reciprocal Rank Fusion, aggregated to documents, as a LangChain
retriever (rebuild phase 2)."""

import asyncio
from collections import defaultdict
from collections.abc import Sequence

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from pydantic import SkipValidation

from customer_agent.application.ports import Clock, KnowledgeSearch
from customer_agent.domain.retrieval import ChunkHit, SearchMode, SearchScope
from customer_agent.observability import step

RRF_K = 60
CANDIDATES_PER_LIST = 20
# Customers paste whole e-mails; the embedding API rejects inputs beyond ~8k tokens and long
# queries dilute relevance. Input guardrails (Day 6) enforce the user-facing limit.
MAX_QUERY_CHARS = 2000

_UMLAUT_TRANSLITERATIONS = {"ae": "ä", "oe": "ö", "ue": "ü"}


def umlaut_variant(text: str) -> str | None:
    """``zurueckschicken`` → ``zurückschicken``. ``None`` if the text contains no candidates.

    Applied only to the lexical query; the original text is always searched as well, so wrong
    transliterations (``Steuer`` → ``Steür``) cannot remove matches.
    """
    variant = text
    for ascii_form, umlaut in _UMLAUT_TRANSLITERATIONS.items():
        variant = variant.replace(ascii_form, umlaut).replace(
            ascii_form.capitalize(), umlaut.upper()
        )
    return variant if variant != text else None


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[ChunkHit]],
    k: int = RRF_K,
    weights: Sequence[float] | None = None,
) -> list[ChunkHit]:
    """Fuse ranked lists by rank only: score = Σ wᵢ / (k + rank). Raw scores are not comparable.

    ``weights`` (one per list, default 1.0) express how much each retriever is trusted.
    """
    weights = weights or [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("one weight per ranked list")
    fused: dict[int, float] = defaultdict(float)
    hits: dict[int, ChunkHit] = {}
    for ranked, weight in zip(ranked_lists, weights, strict=True):
        for rank, hit in enumerate(ranked, start=1):
            fused[hit.chunk_id] += weight / (k + rank)
            known = hits.get(hit.chunk_id)
            if known is None or (known.similarity is None and hit.similarity is not None):
                hits[hit.chunk_id] = hit  # keep the copy that carries the vector similarity
    order = sorted(fused, key=lambda chunk_id: fused[chunk_id], reverse=True)
    return [hits[c].model_copy(update={"score": fused[c]}) for c in order]


def aggregate_by_document(hits: Sequence[ChunkHit]) -> list[Document]:
    """One ``Document`` per knowledge document, scored as its best chunk; its chunks stay
    together as the text for answer generation (D3). ``similarity`` is the best cosine
    similarity, ``None`` if only full-text search found the document."""
    grouped: dict[str, list[ChunkHit]] = defaultdict(list)
    for hit in hits:
        grouped[hit.document_id].append(hit)
    documents = []
    for doc_id, chunks in grouped.items():
        similarities = [c.similarity for c in chunks if c.similarity is not None]
        metadata = {
            "document_id": doc_id,
            "title": chunks[0].title,
            "source_type": chunks[0].source_type.value,
            "trust_level": chunks[0].trust_level.value,
            "score": max(c.score for c in chunks),
            "similarity": max(similarities) if similarities else None,
            "chunk_ids": [c.chunk_id for c in chunks],
        }
        content = "\n\n".join(c.content for c in chunks)
        documents.append(Document(page_content=content, metadata=metadata))
    return sorted(documents, key=lambda d: float(d.metadata["score"]), reverse=True)


class HybridRetriever(BaseRetriever):
    """Full-text and vector search over the knowledge base, fused by rank. The search applies
    the scope filters (status, validity, tenant); ``clock`` and ``tenant_id`` set the scope."""

    search: SkipValidation[KnowledgeSearch]
    embeddings: Embeddings
    clock: SkipValidation[Clock]
    tenant_id: str
    k: int = 5
    mode: SearchMode = SearchMode.HYBRID
    fulltext_weight: float = 1.0

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        raise NotImplementedError("HybridRetriever is async only")

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        query = query.strip()[:MAX_QUERY_CHARS]
        if not query:
            return []
        scope = SearchScope(today=self.clock.today(), tenant_id=self.tenant_id)
        with step("retrieve", mode=self.mode) as span:
            documents = await self._retrieve(query, scope)
            span.set_attribute("documents", [d.metadata["document_id"] for d in documents])
            return documents

    async def _semantic(self, query: str, scope: SearchScope) -> list[ChunkHit]:
        vector = await self.embeddings.aembed_query(query)
        return await self.search.search_vector(vector, scope, CANDIDATES_PER_LIST)

    async def _retrieve(self, query: str, scope: SearchScope) -> list[Document]:
        lexical_query = " ".join(filter(None, [query, umlaut_variant(query)]))
        # Both branches run concurrently; if one fails, the TaskGroup cancels the other.
        try:
            async with asyncio.TaskGroup() as group:
                lexical = (
                    group.create_task(
                        self.search.search_fulltext(lexical_query, scope, CANDIDATES_PER_LIST)
                    )
                    if self.mode in (SearchMode.FULLTEXT, SearchMode.HYBRID)
                    else None
                )
                semantic = (
                    group.create_task(self._semantic(query, scope))
                    if self.mode in (SearchMode.VECTOR, SearchMode.HYBRID)
                    else None
                )
        except ExceptionGroup as errors:
            # Callers handle ProviderUnavailableError (graceful "no answer"); an exception group
            # would slip past them and end as a 500.
            raise errors.exceptions[0] from None
        ranked_lists, weights = [], []
        if lexical is not None:
            ranked_lists.append(lexical.result())
            weights.append(self.fulltext_weight)
        if semantic is not None:
            ranked_lists.append(semantic.result())
            weights.append(1.0)
        fused = reciprocal_rank_fusion(ranked_lists, weights=weights)
        return aggregate_by_document(fused)[: self.k]
