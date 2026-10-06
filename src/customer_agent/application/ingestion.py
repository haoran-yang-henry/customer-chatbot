"""Ingestion use case: synchronise the knowledge index with the approved knowledge documents."""

import hashlib
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from customer_agent.application.chunking import CHUNKER_VERSION, chunk_document
from customer_agent.application.ports import EmbeddingProvider, IndexedDocument, KnowledgeIndex
from customer_agent.domain.knowledge import DocumentStatus, KnowledgeDocument


class IngestionReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    added: list[str]
    updated: list[str]
    unchanged: list[str]
    removed: list[str]
    skipped_not_approved: list[str]
    chunks_embedded: int


def fingerprint(document: KnowledgeDocument, embedding_model: str) -> str:
    """Changes whenever anything that affects the indexed result changes."""
    payload = "\n".join(
        [document.meta.model_dump_json(), document.body, CHUNKER_VERSION, embedding_model]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class IngestionService:
    """Idempotent: a second run with unchanged inputs makes no embedding calls and no writes."""

    def __init__(self, index: KnowledgeIndex, embeddings: EmbeddingProvider) -> None:
        self._index = index
        self._embeddings = embeddings

    async def run(self, documents: Sequence[KnowledgeDocument]) -> IngestionReport:
        approved = [d for d in documents if d.meta.status is DocumentStatus.APPROVED]
        skipped = sorted(
            d.meta.id for d in documents if d.meta.status is not DocumentStatus.APPROVED
        )
        indexed = await self._index.fingerprints()

        model = self._embeddings.model_name
        changed = [d for d in approved if indexed.get(d.meta.id) != fingerprint(d, model)]
        upserts = [await self._prepare(d) for d in changed]
        wanted = {d.meta.id for d in approved}
        removed = sorted(set(indexed) - wanted)

        if upserts or removed:
            await self._index.apply(upserts, removed, embedding_model=model)

        changed_ids = {d.meta.id for d in changed}
        return IngestionReport(
            added=sorted(i for i in changed_ids if i not in indexed),
            updated=sorted(i for i in changed_ids if i in indexed),
            unchanged=sorted(wanted - changed_ids),
            removed=removed,
            skipped_not_approved=skipped,
            chunks_embedded=sum(len(u.chunks) for u in upserts),
        )

    async def _prepare(self, document: KnowledgeDocument) -> IndexedDocument:
        chunks = chunk_document(document)
        vectors = await self._embeddings.embed_documents([c.content for c in chunks])
        return IndexedDocument(
            meta=document.meta,
            fingerprint=fingerprint(document, self._embeddings.model_name),
            chunks=chunks,
            embeddings=vectors,
        )
