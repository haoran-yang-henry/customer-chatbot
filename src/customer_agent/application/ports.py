"""Ports (interfaces) implemented by infrastructure adapters. Added as use cases need them."""

from collections.abc import Sequence
from typing import Protocol, Self

from pydantic import BaseModel, ConfigDict, model_validator

from customer_agent.domain.conversation import AgentResponse, Conversation
from customer_agent.domain.knowledge import KnowledgeChunk, KnowledgeDocumentMeta
from customer_agent.domain.retrieval import ChunkHit, SearchScope


class CustomerServiceAgent(Protocol):
    """Anything that can answer the next turn of a conversation: the real agent or a test double."""

    async def respond(self, conversation: Conversation) -> AgentResponse: ...


class EmbeddingProvider(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


class IndexedDocument(BaseModel):
    """A document ready to be written to the index: metadata, chunks and one embedding per chunk."""

    model_config = ConfigDict(frozen=True)

    meta: KnowledgeDocumentMeta
    fingerprint: str
    chunks: list[KnowledgeChunk]
    embeddings: list[list[float]]

    @model_validator(mode="after")
    def _one_embedding_per_chunk(self) -> Self:
        if len(self.chunks) != len(self.embeddings):
            raise ValueError("each chunk needs exactly one embedding")
        return self


class KnowledgeIndex(Protocol):
    """Write side of the knowledge store, used by ingestion."""

    async def fingerprints(self) -> dict[str, str]:
        """Fingerprint of every indexed document, keyed by document id."""
        ...

    async def apply(
        self, upserts: Sequence[IndexedDocument], deletions: Sequence[str], embedding_model: str
    ) -> None:
        """Replace upserted documents (and their chunks) and delete others, atomically."""
        ...


class KnowledgeSearch(Protocol):
    """Read side: ranked candidate chunks. Implementations must apply the scope filters."""

    async def search_fulltext(
        self, query: str, scope: SearchScope, limit: int
    ) -> list[ChunkHit]: ...

    async def search_vector(
        self, embedding: Sequence[float], scope: SearchScope, limit: int
    ) -> list[ChunkHit]: ...
