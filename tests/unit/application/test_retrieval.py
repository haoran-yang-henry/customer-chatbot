from collections.abc import Sequence
from datetime import date

import pytest

from customer_agent.application.ports import ProviderUnavailableError
from customer_agent.application.retrieval import (
    MAX_QUERY_CHARS,
    RRF_K,
    RetrievalService,
    aggregate_by_document,
    reciprocal_rank_fusion,
    umlaut_variant,
)
from customer_agent.domain.knowledge import SourceType, TrustLevel
from customer_agent.domain.retrieval import ChunkHit, SearchMode, SearchScope
from customer_agent.infrastructure.embeddings import HashingEmbeddingProvider

SCOPE = SearchScope(today=date(2026, 10, 5), tenant_id="nordelektro")


def hit(
    chunk_id: int, document_id: str = "doc", score: float = 0.0, similarity: float | None = None
) -> ChunkHit:
    return ChunkHit(
        chunk_id=chunk_id,
        document_id=document_id,
        title=document_id,
        source_type=SourceType.FAQ,
        trust_level=TrustLevel.OFFICIAL,
        content="…",
        score=score,
        similarity=similarity,
    )


class TestReciprocalRankFusion:
    def test_item_ranked_well_in_both_lists_wins(self) -> None:
        fused = reciprocal_rank_fusion([[hit(1), hit(2)], [hit(2), hit(3)]])

        assert [h.chunk_id for h in fused] == [2, 1, 3]
        assert fused[0].score == pytest.approx(1 / (RRF_K + 2) + 1 / (RRF_K + 1))

    def test_raw_scores_are_ignored(self) -> None:
        fused = reciprocal_rank_fusion([[hit(1, score=0.01), hit(2, score=999.0)]])

        assert [h.chunk_id for h in fused] == [1, 2]

    def test_vector_similarity_survives_fusion(self) -> None:
        lexical, semantic = [hit(1, score=3.2)], [hit(1, score=0.8, similarity=0.8)]

        fused = reciprocal_rank_fusion([lexical, semantic])

        assert fused[0].similarity == 0.8
        assert fused[0].score == pytest.approx(2 / (RRF_K + 1))

    def test_weights_shift_the_fusion_towards_the_trusted_list(self) -> None:
        lexical, semantic = [hit(1), hit(2)], [hit(2), hit(1)]

        equal = reciprocal_rank_fusion([lexical, semantic])
        vector_heavy = reciprocal_rank_fusion([lexical, semantic], weights=[0.5, 1.0])

        assert {h.chunk_id for h in equal[:1]} <= {1, 2}  # tie broken by insertion order
        assert vector_heavy[0].chunk_id == 2

    def test_weights_must_match_lists(self) -> None:
        with pytest.raises(ValueError, match="one weight"):
            reciprocal_rank_fusion([[hit(1)]], weights=[1.0, 1.0])

    def test_empty_input_gives_empty_result(self) -> None:
        assert reciprocal_rank_fusion([[], []]) == []


def test_document_similarity_is_best_chunk_similarity() -> None:
    [document] = aggregate_by_document([hit(1, "a", 0.1, 0.4), hit(2, "a", 0.2, 0.7), hit(3, "a")])

    assert document.similarity == 0.7


def test_documents_score_as_their_best_chunk() -> None:
    documents = aggregate_by_document([hit(1, "a", 0.5), hit(2, "b", 0.9), hit(3, "a", 0.7)])

    assert [(d.document_id, d.score, len(d.chunks)) for d in documents] == [
        ("b", 0.9, 1),
        ("a", 0.7, 2),
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("zurueckschicken", "zurückschicken"),
        ("Oesterreich Gebuehren", "Österreich Gebühren"),
        ("Versand", None),
    ],
)
def test_umlaut_variant(text: str, expected: str | None) -> None:
    assert umlaut_variant(text) == expected


class RecordingSearch:
    def __init__(self) -> None:
        self.fulltext_queries: list[str] = []
        self.vector_calls = 0

    async def search_fulltext(self, query: str, scope: SearchScope, limit: int) -> list[ChunkHit]:
        self.fulltext_queries.append(query)
        return [hit(1, "lexical")]

    async def search_vector(
        self, embedding: Sequence[float], scope: SearchScope, limit: int
    ) -> list[ChunkHit]:
        self.vector_calls += 1
        return [hit(2, "semantic")]


@pytest.mark.parametrize(
    ("mode", "expected_documents"),
    [
        (SearchMode.FULLTEXT, {"lexical"}),
        (SearchMode.VECTOR, {"semantic"}),
        (SearchMode.HYBRID, {"lexical", "semantic"}),
    ],
)
async def test_service_uses_the_lists_of_the_selected_mode(
    mode: SearchMode, expected_documents: set[str]
) -> None:
    service = RetrievalService(RecordingSearch(), HashingEmbeddingProvider(dimensions=8))

    documents = await service.retrieve("Versand", SCOPE, mode=mode)

    assert {d.document_id for d in documents} == expected_documents


async def test_lexical_query_includes_umlaut_variant() -> None:
    search = RecordingSearch()
    service = RetrievalService(search, HashingEmbeddingProvider(dimensions=8))

    await service.retrieve("zurueck", SCOPE, mode=SearchMode.FULLTEXT)

    assert search.fulltext_queries == ["zurueck zurück"]


async def test_blank_query_returns_nothing_without_searching() -> None:
    search = RecordingSearch()
    service = RetrievalService(search, HashingEmbeddingProvider(dimensions=8))

    assert await service.retrieve("   ", SCOPE) == []
    assert search.fulltext_queries == [] and search.vector_calls == 0


async def test_long_query_is_truncated() -> None:
    search = RecordingSearch()
    service = RetrievalService(search, HashingEmbeddingProvider(dimensions=8))

    await service.retrieve("Versand " * 1000, SCOPE, mode=SearchMode.FULLTEXT)

    assert len(search.fulltext_queries[0]) <= MAX_QUERY_CHARS


class DownEmbeddings(HashingEmbeddingProvider):
    async def embed_query(self, text: str) -> list[float]:
        raise ProviderUnavailableError("embedding outage")


async def test_provider_failure_surfaces_as_itself_not_as_an_exception_group() -> None:
    """The orchestrator turns ProviderUnavailableError into a fallback answer; an exception
    group from the concurrent searches would bypass that handler and end as a 500."""
    service = RetrievalService(RecordingSearch(), DownEmbeddings(dimensions=8))

    with pytest.raises(ProviderUnavailableError):
        await service.retrieve("Versand", SCOPE)
