"""Retrieval use case: hybrid search with Reciprocal Rank Fusion, aggregated to documents."""

from collections import defaultdict
from collections.abc import Sequence

from customer_agent.application.ports import EmbeddingProvider, KnowledgeSearch
from customer_agent.domain.retrieval import ChunkHit, RetrievedDocument, SearchMode, SearchScope

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
    ranked_lists: Sequence[Sequence[ChunkHit]], k: int = RRF_K
) -> list[ChunkHit]:
    """Fuse ranked lists by rank only: score = Σ 1 / (k + rank). Raw scores are not comparable."""
    fused: dict[int, float] = defaultdict(float)
    hits: dict[int, ChunkHit] = {}
    for ranked in ranked_lists:
        for rank, hit in enumerate(ranked, start=1):
            fused[hit.chunk_id] += 1.0 / (k + rank)
            known = hits.get(hit.chunk_id)
            if known is None or (known.similarity is None and hit.similarity is not None):
                hits[hit.chunk_id] = hit  # keep the copy that carries the vector similarity
    order = sorted(fused, key=lambda chunk_id: fused[chunk_id], reverse=True)
    return [hits[c].model_copy(update={"score": fused[c]}) for c in order]


def aggregate_by_document(hits: Sequence[ChunkHit]) -> list[RetrievedDocument]:
    """A document scores as its best chunk; chunks stay attached for answer generation (D3)."""
    grouped: dict[str, list[ChunkHit]] = defaultdict(list)
    for hit in hits:
        grouped[hit.document_id].append(hit)
    documents = [
        RetrievedDocument(
            document_id=doc_id,
            title=chunks[0].title,
            source_type=chunks[0].source_type,
            trust_level=chunks[0].trust_level,
            score=max(c.score for c in chunks),
            chunks=chunks,
        )
        for doc_id, chunks in grouped.items()
    ]
    return sorted(documents, key=lambda d: d.score, reverse=True)


class RetrievalService:
    def __init__(self, search: KnowledgeSearch, embeddings: EmbeddingProvider) -> None:
        self._search = search
        self._embeddings = embeddings

    async def retrieve(
        self,
        query: str,
        scope: SearchScope,
        *,
        top_k: int = 5,
        mode: SearchMode = SearchMode.HYBRID,
    ) -> list[RetrievedDocument]:
        query = query.strip()[:MAX_QUERY_CHARS]
        if not query:
            return []
        ranked_lists: list[list[ChunkHit]] = []
        if mode in (SearchMode.FULLTEXT, SearchMode.HYBRID):
            lexical_query = " ".join(filter(None, [query, umlaut_variant(query)]))
            ranked_lists.append(
                await self._search.search_fulltext(lexical_query, scope, CANDIDATES_PER_LIST)
            )
        if mode in (SearchMode.VECTOR, SearchMode.HYBRID):
            vector = await self._embeddings.embed_query(query)
            ranked_lists.append(
                await self._search.search_vector(vector, scope, CANDIDATES_PER_LIST)
            )
        return aggregate_by_document(reciprocal_rank_fusion(ranked_lists))[:top_k]
