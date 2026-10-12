"""Ports (interfaces) implemented by infrastructure adapters. Added as use cases need them."""

from collections.abc import AsyncIterator, Sequence
from datetime import date
from typing import Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, model_validator

from customer_agent.domain.conversation import AgentResponse, Conversation, Message
from customer_agent.domain.knowledge import KnowledgeChunk, KnowledgeDocumentMeta
from customer_agent.domain.retrieval import (
    ChunkHit,
    RetrievedDocument,
    SearchMode,
    SearchScope,
)
from customer_agent.domain.session import SessionState


class Principal(BaseModel):
    """The authenticated customer, taken only from a verified token (FR-20, threat model T13)."""

    model_config = ConfigDict(frozen=True)

    subject: str
    customer_id: str
    tenant_id: str


class AuthenticationError(Exception):
    """The presented token is missing required claims, expired, forged or for another audience."""


class TokenVerifier(Protocol):
    async def verify(self, token: str) -> Principal: ...


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


class ProviderUnavailableError(RuntimeError):
    """An external model provider failed (timeout, 5xx, auth). Adapters raise this instead of
    SDK-specific exceptions so the application layer stays provider-agnostic."""


class Clock(Protocol):
    def today(self) -> date: ...


class Retriever(Protocol):
    async def retrieve(
        self,
        query: str,
        scope: SearchScope,
        *,
        top_k: int = 5,
        mode: SearchMode = SearchMode.HYBRID,
    ) -> list[RetrievedDocument]: ...


class ChatMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str


class LLMResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class LLMProvider(Protocol):
    @property
    def model_name(self) -> str: ...

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
        session_id: str | None = None,
    ) -> LLMResult:
        """``session_id`` is stable per conversation (routing / prompt caching at the provider)."""
        ...

    def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
        session_id: str | None = None,
    ) -> AsyncIterator[str]: ...


class StoredSession(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    tenant_id: str
    customer_id: str | None
    state: SessionState
    messages: list[Message]
    version: int = 0  # optimistic concurrency token, incremented by every saved turn


class ConcurrentTurnError(RuntimeError):
    """Another turn of the same session was saved since this one loaded it (lost update)."""


class SessionStore(Protocol):
    async def create(self, *, tenant_id: str, customer_id: str | None) -> StoredSession: ...

    async def load(self, session_id: str) -> StoredSession | None: ...

    async def save_turn(
        self,
        session_id: str,
        *,
        expected_version: int,
        state: SessionState,
        user: str,
        reply: AgentResponse,
    ) -> None:
        """Append the user message and the reply and store the new state, atomically.

        Raises ``ConcurrentTurnError`` if the stored version differs from ``expected_version``.
        """
        ...
