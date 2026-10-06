"""Retrieval under production conditions, with the real embedding model.

Run with ``make test-llm`` (``--embeddings=openai``). Skipped otherwise.

The customer phrasings below are deliberately **not** in ``eval/datasets/core.jsonl``: they act as
a held-out check that retrieval quality is not tuned to the evaluation set.
"""

import asyncio
import shutil
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import httpx2
import openai
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from customer_agent.application.ingestion import IngestionService
from customer_agent.application.ports import EmbeddingProvider, ProviderUnavailableError
from customer_agent.application.retrieval import RetrievalService
from customer_agent.config import Settings
from customer_agent.domain.retrieval import SearchMode, SearchScope
from customer_agent.infrastructure.db.knowledge_repository import PostgresKnowledgeRepository
from customer_agent.infrastructure.embeddings import OpenAIEmbeddingProvider
from customer_agent.infrastructure.knowledge_files import load_knowledge_directory
from tests.conftest import KNOWLEDGE_DIR, REFERENCE_DATE, TENANT_ID

pytestmark = pytest.mark.llm

SCOPE = SearchScope(today=REFERENCE_DATE, tenant_id=TENANT_ID)


class CountingEmbeddings:
    """Wraps the real provider to count API usage; optionally fails on the n-th batch."""

    def __init__(self, inner: EmbeddingProvider, fail_on_batch: int | None = None) -> None:
        self._inner = inner
        self._fail_on_batch = fail_on_batch
        self.batches = 0
        self.texts = 0

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.batches += 1
        if self.batches == self._fail_on_batch:
            raise openai.APIConnectionError(
                request=httpx2.Request("POST", "https://provider/embeddings")
            )
        self.texts += len(texts)
        return await self._inner.embed_documents(texts)

    async def embed_query(self, text: str) -> list[float]:
        return await self._inner.embed_query(text)


@pytest.fixture
async def service(
    repository: PostgresKnowledgeRepository, embeddings: EmbeddingProvider
) -> RetrievalService:
    await IngestionService(repository, embeddings).run(load_knowledge_directory(KNOWLEDGE_DIR))
    return RetrievalService(repository, embeddings)


async def ids(
    service: RetrievalService, query: str, scope: SearchScope = SCOPE, **kw: object
) -> list[str]:
    return [d.document_id for d in await service.retrieve(query, scope, **kw)]  # type: ignore[arg-type]


# --- 1. How real customers phrase things --------------------------------------------------

REAL_PHRASINGS = [
    ("Muss ich für die Retoure was zahlen?", {"policy-returns-v5"}),
    ("Bis wann kann ich das Teil noch zurückbringen?", {"policy-returns-v5"}),
    ("retoure kostn?", {"policy-returns-v5"}),
    ("Rücksendekosten Österreich", {"policy-returns-v5"}),
    ("kann ich mit klarna zahlen", {"info-payment-methods"}),
    ("Was passiert, wenn mein Paket kaputt ankommt?", {"faq-delivery-problems"}),
    ("Wie lange dauert es, bis das Geld wieder auf meinem Konto ist?", {"faq-refund-timeline"}),
    ("Schickt ihr auch nach Wien?", {"info-shipping-international"}),
    ("Grüezi, was kostet de Versand i d'Schwiiz?", {"info-shipping-international"}),
    ("Do you ship to Austria?", {"info-shipping-international"}),
    ("Hält der Akku von den Over-Ear-Kopfhörern lange?", {"product-klangwerk-anc-700"}),
    ("Kann ich mit der Uhr schwimmen gehen?", {"product-velto-watch-3"}),
    ("Passwort vergessen, was tun?", {"faq-account"}),
    ("Ich hätte gern eine Rechnung für meine Firma", {"faq-invoice"}),
    ("Mein Laptop ist nach 8 Monaten kaputt gegangen", {"policy-warranty"}),
    ("Hotline Öffnungszeiten Samstag", {"info-customer-service-contact"}),
    ("Wie storniere ich meine Bestellung?", {"policy-order-cancellation"}),
    ("Ich will meine Daten löschen lassen (DSGVO)", {"policy-privacy", "faq-account"}),
    (
        "📦❓ wo bleibt mein paket??? schon 5 tage!!!",
        {"faq-delivery-problems", "info-shipping-options"},
    ),
]


@pytest.mark.parametrize(
    ("query", "acceptable"), REAL_PHRASINGS, ids=[q for q, _ in REAL_PHRASINGS]
)
async def test_real_customer_phrasing_finds_the_right_document_in_top_5(
    service: RetrievalService, query: str, acceptable: set[str]
) -> None:
    top5 = await ids(service, query, top_k=5)

    assert acceptable & set(top5), f"{query!r} → {top5}"


# --- 2. Semantic closeness must never bypass the filters ----------------------------------
# Real embeddings rank these forbidden documents at the very top if filters were missing.

FORBIDDEN_NEIGHBOURS = [
    (
        "Bis zu welchem Betrag dürfen Mitarbeitende Kulanzgutscheine ausstellen?",
        "internal-goodwill-guidelines",
    ),
    ("Gibt es die 60 Tage Rückgabe für Plus-Mitglieder schon?", "policy-returns-v6"),
    ("Rücksendung kostet 5,95 € und geht nur 14 Tage, oder?", "policy-returns-v4"),
    ("20 % Rabatt auf Klangwerk-Kopfhörer im Winter-Sale", "promo-winter-sale-2025"),
]


@pytest.mark.parametrize("mode", list(SearchMode))
@pytest.mark.parametrize(
    ("query", "forbidden"), FORBIDDEN_NEIGHBOURS, ids=[f for _, f in FORBIDDEN_NEIGHBOURS]
)
async def test_semantically_closest_forbidden_document_is_never_returned(
    service: RetrievalService, query: str, forbidden: str, mode: SearchMode
) -> None:
    assert forbidden not in await ids(service, query, top_k=50, mode=mode)


async def test_poisoned_community_post_is_retrievable_but_flagged_untrusted(
    service: RetrievalService,
) -> None:
    documents = await service.retrieve(
        "Velto Watch verbindet sich nicht mit dem Handy", SCOPE, top_k=10
    )
    by_id = {d.document_id: d for d in documents}

    assert "product-velto-watch-3" in by_id
    if "community-velto-watch-pairing-tip" in by_id:
        assert by_id["community-velto-watch-pairing-tip"].trust_level == "untrusted"


# --- 3. Relevance signal for "no answer" decisions (input for Day 3) ----------------------

ON_TOPIC = [
    "Was kostet der Expressversand?",
    "Kann ich auf Rechnung bezahlen?",
    "Wie lange habe ich Gewährleistung?",
]
OFF_TOPIC = [
    "Wie wird das Wetter morgen in Berlin?",
    "Schreib mir ein Gedicht über den Herbst",
    "Soll ich Bitcoin kaufen?",
]


async def test_off_topic_queries_score_lower_than_on_topic_queries(
    service: RetrievalService,
) -> None:
    async def best_vector_score(query: str) -> float:
        documents = await service.retrieve(query, SCOPE, top_k=1, mode=SearchMode.VECTOR)
        similarity = documents[0].similarity if documents else None
        return similarity or 0.0

    # Vector search always returns *something*; only the score separates relevant from irrelevant.
    on = [await best_vector_score(q) for q in ON_TOPIC]
    off = [await best_vector_score(q) for q in OFF_TOPIC]

    assert min(on) > max(off), f"on-topic {on} vs off-topic {off}"


# --- 4. Knowledge lifecycle as editors would run it -----------------------------------------


def _set_front_matter(path: Path, **fields: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        key = line.split(":", 1)[0]
        if key in fields:
            lines[i] = f"{key}: {fields[key]}"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def knowledge_copy(tmp_path: Path) -> Path:
    target = tmp_path / "knowledge"
    shutil.copytree(KNOWLEDGE_DIR, target)
    return target


async def test_scheduled_policy_version_switches_on_its_valid_from_date(
    repository: PostgresKnowledgeRepository, embeddings: EmbeddingProvider, knowledge_copy: Path
) -> None:
    _set_front_matter(
        knowledge_copy / "policy-returns-v6.md",
        status="approved",
        acl="[public]",
        trust_level="official",
    )
    _set_front_matter(knowledge_copy / "policy-returns-v5.md", valid_until="2026-12-31")
    await IngestionService(repository, embeddings).run(load_knowledge_directory(knowledge_copy))
    service = RetrievalService(repository, embeddings)
    query = "Wie lange habe ich Zeit für eine Rückgabe?"

    before = await ids(
        service, query, SearchScope(today=date(2026, 12, 31), tenant_id=TENANT_ID), top_k=10
    )
    after = await ids(
        service, query, SearchScope(today=date(2027, 1, 1), tenant_id=TENANT_ID), top_k=10
    )

    assert "policy-returns-v5" in before and "policy-returns-v6" not in before
    assert "policy-returns-v6" in after and "policy-returns-v5" not in after


async def test_editing_one_document_re_embeds_only_that_document(
    repository: PostgresKnowledgeRepository, embeddings: EmbeddingProvider, knowledge_copy: Path
) -> None:
    counting = CountingEmbeddings(embeddings)
    service = IngestionService(repository, counting)
    await service.run(load_knowledge_directory(knowledge_copy))
    texts_after_first_run = counting.texts

    path = knowledge_copy / "info-shipping-costs.md"
    path.write_text(path.read_text(encoding="utf-8").replace("4,95 €", "5,49 €"), encoding="utf-8")
    report = await service.run(load_knowledge_directory(knowledge_copy))

    assert report.updated == ["info-shipping-costs"]
    assert counting.texts - texts_after_first_run == report.chunks_embedded == 1


async def test_archiving_removes_the_document_from_the_index(
    repository: PostgresKnowledgeRepository, embeddings: EmbeddingProvider, knowledge_copy: Path
) -> None:
    service = IngestionService(repository, embeddings)
    await service.run(load_knowledge_directory(knowledge_copy))

    _set_front_matter(knowledge_copy / "faq-invoice.md", status="archived")
    report = await service.run(load_knowledge_directory(knowledge_copy))

    assert report.removed == ["faq-invoice"]
    assert "faq-invoice" not in await repository.fingerprints()


# --- 5. Failures and hostile input ------------------------------------------------------------


async def test_provider_failure_mid_ingestion_leaves_index_untouched(
    repository: PostgresKnowledgeRepository, embeddings: EmbeddingProvider
) -> None:
    failing = CountingEmbeddings(embeddings, fail_on_batch=5)

    with pytest.raises(openai.APIConnectionError):
        await IngestionService(repository, failing).run(load_knowledge_directory(KNOWLEDGE_DIR))

    assert await repository.fingerprints() == {}


async def test_invalid_api_key_raises_a_provider_error() -> None:
    settings = Settings()
    client = openai.AsyncOpenAI(
        api_key="sk-invalid", base_url=settings.openai_base_url, max_retries=0
    )
    provider = OpenAIEmbeddingProvider(
        client, settings.embedding_model, settings.embedding_dimensions
    )

    with pytest.raises(ProviderUnavailableError):
        await provider.embed_query("Versand")


@pytest.mark.parametrize(
    "query",
    [
        "'; DROP TABLE knowledge_chunks; --",
        "Versand & Kosten | !(Express) <-> :*",
        "O'Reilly's Kopfhörer \\ \"Test\"",
        "",
        "   \n\t ",
        "Versandkosten " * 3000,
    ],
    ids=["sql-injection", "tsquery-operators", "quotes", "empty", "whitespace", "very-long"],
)
async def test_hostile_or_malformed_queries_do_not_break_search(
    service: RetrievalService, engine: AsyncEngine, query: str
) -> None:
    await service.retrieve(query, SCOPE)

    async with engine.connect() as connection:
        count = await connection.scalar(text("SELECT count(*) FROM knowledge_chunks"))
    assert count and count > 0


async def test_concurrent_queries(service: RetrievalService) -> None:
    queries = [q for q, _ in REAL_PHRASINGS] + ON_TOPIC

    results = await asyncio.gather(*(service.retrieve(q, SCOPE) for q in queries))

    assert all(results)
