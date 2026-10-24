from collections.abc import Sequence

from customer_agent.application.ingestion import IngestionService, fingerprint
from customer_agent.application.ports import IndexedDocument
from tests.factories import make_document
from tests.fakes import HashingEmbeddings


class InMemoryIndex:
    def __init__(self) -> None:
        self.documents: dict[str, IndexedDocument] = {}
        self.apply_calls = 0

    async def fingerprints(self) -> dict[str, str]:
        return {doc_id: doc.fingerprint for doc_id, doc in self.documents.items()}

    async def apply(
        self, upserts: Sequence[IndexedDocument], deletions: Sequence[str], embedding_model: str
    ) -> None:
        self.apply_calls += 1
        for doc_id in deletions:
            del self.documents[doc_id]
        for doc in upserts:
            self.documents[doc.meta.id] = doc


class CountingEmbeddings(HashingEmbeddings):
    def __init__(self, model_name: str = "fake-v1") -> None:
        super().__init__(dimensions=8, model_name=model_name)
        self.texts_embedded = 0

    async def aembed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.texts_embedded += len(texts)
        return await super().aembed_documents(texts)


DOCS = [
    make_document(id="faq-a"),
    make_document(id="faq-b"),
    make_document(id="faq-draft", status="draft"),
]


async def test_first_run_indexes_only_approved_documents() -> None:
    index = InMemoryIndex()

    report = await IngestionService(index, CountingEmbeddings(), embedding_model="fake-v1").run(
        DOCS
    )

    assert report.added == ["faq-a", "faq-b"]
    assert report.skipped_not_approved == ["faq-draft"]
    assert set(index.documents) == {"faq-a", "faq-b"}


async def test_second_run_is_a_no_op() -> None:
    index, embeddings = InMemoryIndex(), CountingEmbeddings()
    service = IngestionService(index, embeddings, embedding_model=embeddings.model_name)
    await service.run(DOCS)
    embedded_before = embeddings.texts_embedded

    report = await service.run(DOCS)

    assert report.unchanged == ["faq-a", "faq-b"]
    assert embeddings.texts_embedded == embedded_before
    assert index.apply_calls == 1


async def test_changed_body_is_reindexed_and_missing_document_removed() -> None:
    index = InMemoryIndex()
    service = IngestionService(index, CountingEmbeddings(), embedding_model="fake-v1")
    await service.run(DOCS)

    report = await service.run([make_document(id="faq-a", body="Neuer Inhalt.")])

    assert report.updated == ["faq-a"]
    assert report.removed == ["faq-b"]
    assert set(index.documents) == {"faq-a"}


async def test_new_embedding_model_reindexes_everything() -> None:
    index = InMemoryIndex()
    await IngestionService(index, CountingEmbeddings("model-1"), embedding_model="model-1").run(
        DOCS
    )

    report = await IngestionService(
        index, CountingEmbeddings("model-2"), embedding_model="model-2"
    ).run(DOCS)

    assert report.updated == ["faq-a", "faq-b"]


def test_fingerprint_depends_on_metadata() -> None:
    assert fingerprint(make_document(), "m") != fingerprint(make_document(version=2), "m")
