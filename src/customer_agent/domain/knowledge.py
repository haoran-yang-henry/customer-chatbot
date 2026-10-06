"""Knowledge document metadata and retrieval eligibility rules (docs/architecture.md §6.1)."""

from datetime import date
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from customer_agent.domain.intents import Intent


class SourceType(StrEnum):
    POLICY = "policy"
    FAQ = "faq"
    PRODUCT = "product"
    PROMOTION = "promotion"
    INTERNAL = "internal"
    COMMUNITY = "community"


class DocumentStatus(StrEnum):
    DRAFT = "draft"
    APPROVED = "approved"
    ARCHIVED = "archived"


class TrustLevel(StrEnum):
    OFFICIAL = "official"
    INTERNAL = "internal"
    UNTRUSTED = "untrusted"


class AccessLevel(StrEnum):
    PUBLIC = "public"
    INTERNAL = "internal"


class KnowledgeDocumentMeta(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")
    title: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    intents: list[Intent] = Field(min_length=1)
    source_type: SourceType
    language: Literal["de"]
    version: int = Field(ge=1)
    valid_from: date
    valid_until: date | None
    status: DocumentStatus
    trust_level: TrustLevel
    tenant_id: str = Field(min_length=1)
    acl: list[AccessLevel] = Field(min_length=1)
    owner: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_validity_window(self) -> Self:
        if self.valid_until is not None and self.valid_until < self.valid_from:
            raise ValueError("valid_until must not be before valid_from")
        return self

    def is_active_on(self, day: date) -> bool:
        """Approved and inside its (inclusive) validity window."""
        if self.status is not DocumentStatus.APPROVED or day < self.valid_from:
            return False
        return self.valid_until is None or day <= self.valid_until

    def is_retrievable_for_customer(self, day: date, tenant_id: str) -> bool:
        """May this document be shown to a customer of ``tenant_id`` on ``day``?"""
        return (
            self.is_active_on(day)
            and self.tenant_id == tenant_id
            and AccessLevel.PUBLIC in self.acl
        )


class KnowledgeDocument(BaseModel):
    model_config = ConfigDict(frozen=True)

    meta: KnowledgeDocumentMeta
    body: str = Field(min_length=1)


class KnowledgeChunk(BaseModel):
    """A retrievable section of a document; ``content`` carries the document title as context."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    position: int = Field(ge=0)
    heading: str | None
    content: str = Field(min_length=1)
