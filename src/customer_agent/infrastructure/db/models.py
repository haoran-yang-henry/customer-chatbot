"""ORM models. The schema itself is owned by the Alembic migrations in ``migrations/``."""

from datetime import date, datetime
from decimal import Decimal
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Computed,
    DateTime,
    ForeignKey,
    Identity,
    Index,
    Numeric,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Fixed by the migration; a different embedding dimension needs a new migration (ADR-0007).
EMBEDDING_DIMENSIONS = 1536


class Base(DeclarativeBase):
    type_annotation_map = {str: Text}  # noqa: RUF012 - SQLAlchemy reads this class attribute


class KnowledgeDocumentRow(Base):
    __tablename__ = "knowledge_documents"

    id: Mapped[str] = mapped_column(primary_key=True)
    title: Mapped[str]
    topic: Mapped[str]
    source_type: Mapped[str]
    trust_level: Mapped[str]
    status: Mapped[str]
    version: Mapped[int]
    valid_from: Mapped[date]
    valid_until: Mapped[date | None]
    tenant_id: Mapped[str]
    acl: Mapped[list[str]] = mapped_column(ARRAY(Text))
    intents: Mapped[list[str]] = mapped_column(ARRAY(Text))
    owner: Mapped[str]
    fingerprint: Mapped[str]
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    chunks: Mapped[list["KnowledgeChunkRow"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )


class KnowledgeChunkRow(Base):
    __tablename__ = "knowledge_chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "position"),
        Index("ix_knowledge_chunks_fts", "fts", postgresql_using="gin"),
        Index(
            "ix_knowledge_chunks_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_documents.id", ondelete="CASCADE"), index=True
    )
    position: Mapped[int]
    heading: Mapped[str | None]
    content: Mapped[str]
    fts: Mapped[str] = mapped_column(
        TSVECTOR, Computed("to_tsvector('german'::regconfig, content)", persisted=True)
    )
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSIONS))
    embedding_model: Mapped[str]

    document: Mapped[KnowledgeDocumentRow] = relationship(back_populates="chunks")


class ChatSessionRow(Base):
    """Ownership and the turn lease; transcript and state live in the checkpoint tables."""

    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(primary_key=True)
    tenant_id: Mapped[str]
    customer_id: Mapped[str | None]
    turn_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class CustomerRow(Base):
    __tablename__ = "customers"

    customer_id: Mapped[str] = mapped_column(primary_key=True)
    tenant_id: Mapped[str]
    first_name: Mapped[str]
    last_name: Mapped[str]
    email: Mapped[str]
    country: Mapped[str]


class OrderRow(Base):
    __tablename__ = "orders"

    order_id: Mapped[str] = mapped_column(primary_key=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.customer_id"), index=True)
    tenant_id: Mapped[str]
    status: Mapped[str]
    created_at: Mapped[date]
    items: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)
    total: Mapped[Decimal] = mapped_column(Numeric(10, 2))
    payment_method: Mapped[str]
    payment_status: Mapped[str]
    shipping: Mapped[dict[str, Any]] = mapped_column(JSONB)
    return_info: Mapped[dict[str, Any] | None] = mapped_column(JSONB)


class TicketRow(Base):
    __tablename__ = "tickets"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    tenant_id: Mapped[str]
    customer_id: Mapped[str | None]
    session_id: Mapped[str]
    intent: Mapped[str | None]
    order_id: Mapped[str | None]
    summary: Mapped[str]
    status: Mapped[str] = mapped_column(server_default="open")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AuditLogRow(Base):
    __tablename__ = "audit_log"
    __table_args__ = (Index("ix_audit_log_customer_at", "customer_id", "at"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    request_id: Mapped[str]
    session_id: Mapped[str]
    customer_id: Mapped[str | None]
    tool: Mapped[str]
    arguments: Mapped[dict[str, str]] = mapped_column(JSONB)
    outcome: Mapped[str]
