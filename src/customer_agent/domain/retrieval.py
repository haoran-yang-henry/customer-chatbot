"""Retrieval value objects shared by the retrieval use case, adapters and evaluation."""

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from customer_agent.domain.knowledge import SourceType, TrustLevel


class SearchMode(StrEnum):
    FULLTEXT = "fulltext"
    VECTOR = "vector"
    HYBRID = "hybrid"


class SearchScope(BaseModel):
    """Who is searching and when: drives the mandatory status/validity/tenant/ACL filters."""

    model_config = ConfigDict(frozen=True)

    today: date
    tenant_id: str


class ChunkHit(BaseModel):
    """One chunk in a ranked candidate list.

    ``score`` ranks within one list (after fusion: the RRF score) and is not an absolute measure.
    ``similarity`` is the cosine similarity to the query when the vector search produced this
    chunk; it is the absolute relevance signal and survives fusion.
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: int
    document_id: str
    title: str
    source_type: SourceType
    trust_level: TrustLevel
    content: str
    score: float
    similarity: float | None = None


class RetrievedDocument(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    title: str
    source_type: SourceType
    trust_level: TrustLevel
    score: float
    chunks: list[ChunkHit] = Field(min_length=1)

    @property
    def similarity(self) -> float | None:
        """Best cosine similarity among the chunks; ``None`` if only full-text search found it."""
        values = [c.similarity for c in self.chunks if c.similarity is not None]
        return max(values) if values else None
