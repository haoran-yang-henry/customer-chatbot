"""Retrieval value objects shared by the retriever and the search adapters."""

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

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
