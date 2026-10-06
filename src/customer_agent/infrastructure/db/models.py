"""ORM models. The schema itself is owned by the Alembic migrations in ``migrations/``."""

from datetime import date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Computed,
    DateTime,
    ForeignKey,
    Identity,
    Index,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, TSVECTOR
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
