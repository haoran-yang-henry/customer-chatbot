"""Ingestion and search mechanics against a real Postgres with the real knowledge base.

Embedder-agnostic: passes with the hashing embedder (default) and with the real model
(``--embeddings=openai``). Semantic behaviour is covered in ``test_retrieval_production.py``.
"""

from datetime import date

import pytest

from customer_agent.application.ingestion import IngestionService
from customer_agent.application.ports import EmbeddingProvider
from customer_agent.application.retrieval import RetrievalService
from customer_agent.domain.retrieval import SearchMode, SearchScope
from customer_agent.infrastructure.db.knowledge_repository import PostgresKnowledgeRepository
from customer_agent.infrastructure.knowledge_files import load_knowledge_directory
from tests.conftest import KNOWLEDGE_DIR, REFERENCE_DATE, TENANT_ID

SCOPE = SearchScope(today=REFERENCE_DATE, tenant_id=TENANT_ID)
NEVER_RETRIEVABLE = {
    "policy-returns-v4",  # archived
    "policy-returns-v6",  # draft
    "internal-goodwill-guidelines",  # internal ACL
    "promo-winter-sale-2025",  # expired on the reference date
}


@pytest.fixture
async def ingested(
    repository: PostgresKnowledgeRepository, embeddings: EmbeddingProvider
) -> RetrievalService:
    await IngestionService(repository, embeddings).run(load_knowledge_directory(KNOWLEDGE_DIR))
    return RetrievalService(repository, embeddings)


async def test_ingestion_is_idempotent(
    repository: PostgresKnowledgeRepository, embeddings: EmbeddingProvider
) -> None:
    service = IngestionService(repository, embeddings)
    documents = load_knowledge_directory(KNOWLEDGE_DIR)

    first = await service.run(documents)
    second = await service.run(documents)

    assert first.added and not first.updated
    assert "policy-returns-v6" in first.skipped_not_approved
    assert second.unchanged == sorted(first.added)
    assert not (second.added or second.updated or second.removed)
    assert second.chunks_embedded == 0


async def test_fulltext_matches_german_inflections(ingested: RetrievalService) -> None:
    # "Rücksendungen" (plural) must match "Rücksendung" via the german stemmer.
    documents = await ingested.retrieve("Rücksendungen kostenlos", SCOPE, mode=SearchMode.FULLTEXT)

    assert "policy-returns-v5" in [d.document_id for d in documents]


async def test_fulltext_finds_query_without_umlauts(ingested: RetrievalService) -> None:
    documents = await ingested.retrieve("Rueckgabe Hygienesiegel", SCOPE, mode=SearchMode.FULLTEXT)

    assert documents[0].document_id == "policy-return-exceptions"


async def test_stop_words_only_query_returns_nothing_from_fulltext(
    ingested: RetrievalService,
) -> None:
    assert await ingested.retrieve("und die der", SCOPE, mode=SearchMode.FULLTEXT) == []


@pytest.mark.parametrize("mode", list(SearchMode))
async def test_forbidden_documents_are_never_returned(
    ingested: RetrievalService, mode: SearchMode
) -> None:
    queries = [
        "Rückgaberecht 60 Tage",
        "Rückgabe 14 Tage 5,95 € Rücksendekosten",
        "Kulanzgutschein Teamleitung Eskalationscode",
        "Winter-Sale Rabatt Kopfhörer",
    ]
    for query in queries:
        documents = await ingested.retrieve(query, SCOPE, top_k=50, mode=mode)
        assert not NEVER_RETRIEVABLE & {d.document_id for d in documents}, query


async def test_validity_window_is_evaluated_at_query_time(ingested: RetrievalService) -> None:
    during_sale = SearchScope(today=date(2025, 12, 1), tenant_id=TENANT_ID)

    documents = await ingested.retrieve("Winter-Sale Rabatt", during_sale, mode=SearchMode.FULLTEXT)

    assert "promo-winter-sale-2025" in [d.document_id for d in documents]


async def test_other_tenant_sees_nothing(ingested: RetrievalService) -> None:
    other = SearchScope(today=REFERENCE_DATE, tenant_id="other-shop")

    assert await ingested.retrieve("Versand", other, mode=SearchMode.HYBRID) == []


async def test_untrusted_documents_are_returned_with_their_trust_level(
    ingested: RetrievalService,
) -> None:
    documents = await ingested.retrieve(
        "Velto Watch verbindet sich nicht", SCOPE, top_k=10, mode=SearchMode.FULLTEXT
    )
    community = [d for d in documents if d.document_id == "community-velto-watch-pairing-tip"]

    assert community and community[0].trust_level == "untrusted"
