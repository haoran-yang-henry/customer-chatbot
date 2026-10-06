"""PostgreSQL implementation of the ``KnowledgeIndex`` and ``KnowledgeSearch`` ports."""

from collections.abc import Sequence
from typing import Any

from sqlalchemy import ColumnElement, Row, delete, func, literal, literal_column, or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from customer_agent.application.ports import IndexedDocument
from customer_agent.domain.knowledge import AccessLevel, DocumentStatus, SourceType, TrustLevel
from customer_agent.domain.retrieval import ChunkHit, SearchScope
from customer_agent.infrastructure.db.models import KnowledgeChunkRow, KnowledgeDocumentRow

_Doc = KnowledgeDocumentRow
_Chunk = KnowledgeChunkRow


def _scope_filters(scope: SearchScope) -> list[ColumnElement[bool]]:
    """The mandatory retrieval filters (architecture §6.1). Every search query applies all."""
    return [
        _Doc.status == DocumentStatus.APPROVED.value,
        _Doc.valid_from <= scope.today,
        or_(_Doc.valid_until.is_(None), _Doc.valid_until >= scope.today),
        _Doc.tenant_id == scope.tenant_id,
        _Doc.acl.contains([AccessLevel.PUBLIC.value]),
    ]


def _any_term_tsquery(query: str) -> ColumnElement[Any]:
    """OR-combine the German lexemes of ``query``.

    ``websearch_to_tsquery`` would AND all terms, so a natural-language question rarely matches
    any single chunk. Instead: stem with the ``german`` config, then join lexemes with ``|`` and
    let ``ts_rank_cd`` reward chunks that match more of them. Lexemes are already stemmed, so they
    are parsed with the ``simple`` config to avoid stemming twice.
    """
    lexemes = func.unnest(
        func.to_tsvector(literal_column("'german'::regconfig"), query)
    ).table_valued("lexeme", "positions", "weights")
    joined = select(func.string_agg(func.quote_literal(lexemes.c.lexeme), literal(" | ")))
    return func.to_tsquery(literal_column("'simple'::regconfig"), joined.scalar_subquery())


def _to_hit(row: Row[Any], *, is_similarity: bool = False) -> ChunkHit:
    return ChunkHit(
        chunk_id=row.id,
        document_id=row.document_id,
        title=row.title,
        source_type=SourceType(row.source_type),
        trust_level=TrustLevel(row.trust_level),
        content=row.content,
        score=float(row.score),
        similarity=float(row.score) if is_similarity else None,
    )


class PostgresKnowledgeRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    # --- KnowledgeIndex -------------------------------------------------------------------

    async def fingerprints(self) -> dict[str, str]:
        async with self._sessions() as session:
            rows = await session.execute(select(_Doc.id, _Doc.fingerprint))
            return {row.id: row.fingerprint for row in rows}

    async def apply(
        self, upserts: Sequence[IndexedDocument], deletions: Sequence[str], embedding_model: str
    ) -> None:
        """Delete-then-insert per document inside one transaction (chunks cascade)."""
        replaced = [doc.meta.id for doc in upserts]
        async with self._sessions.begin() as session:
            if replaced or deletions:
                await session.execute(delete(_Doc).where(_Doc.id.in_([*replaced, *deletions])))
            session.add_all(self._to_row(doc, embedding_model) for doc in upserts)

    @staticmethod
    def _to_row(doc: IndexedDocument, embedding_model: str) -> KnowledgeDocumentRow:
        meta = doc.meta
        return KnowledgeDocumentRow(
            id=meta.id,
            title=meta.title,
            topic=meta.topic,
            source_type=meta.source_type.value,
            trust_level=meta.trust_level.value,
            status=meta.status.value,
            version=meta.version,
            valid_from=meta.valid_from,
            valid_until=meta.valid_until,
            tenant_id=meta.tenant_id,
            acl=[a.value for a in meta.acl],
            intents=[i.value for i in meta.intents],
            owner=meta.owner,
            fingerprint=doc.fingerprint,
            chunks=[
                KnowledgeChunkRow(
                    position=chunk.position,
                    heading=chunk.heading,
                    content=chunk.content,
                    embedding=list(vector),
                    embedding_model=embedding_model,
                )
                for chunk, vector in zip(doc.chunks, doc.embeddings, strict=True)
            ],
        )

    # --- KnowledgeSearch ------------------------------------------------------------------

    def _hit_columns(self, score: ColumnElement[Any]) -> tuple[Any, ...]:
        return (
            _Chunk.id,
            _Chunk.document_id,
            _Doc.title,
            _Doc.source_type,
            _Doc.trust_level,
            _Chunk.content,
            score.label("score"),
        )

    async def search_fulltext(self, query: str, scope: SearchScope, limit: int) -> list[ChunkHit]:
        tsquery = _any_term_tsquery(query)
        rank = func.ts_rank_cd(_Chunk.fts, tsquery)
        stmt = (
            select(*self._hit_columns(rank))
            .join(_Doc, _Chunk.document_id == _Doc.id)
            .where(*_scope_filters(scope), _Chunk.fts.bool_op("@@")(tsquery))
            .order_by(rank.desc(), _Chunk.id)
            .limit(limit)
        )
        async with self._sessions() as session:
            return [_to_hit(row) for row in await session.execute(stmt)]

    async def search_vector(
        self, embedding: Sequence[float], scope: SearchScope, limit: int
    ) -> list[ChunkHit]:
        distance = _Chunk.embedding.cosine_distance(list(embedding))
        stmt = (
            select(*self._hit_columns(1 - distance))
            .join(_Doc, _Chunk.document_id == _Doc.id)
            .where(*_scope_filters(scope))
            .order_by(distance, _Chunk.id)
            .limit(limit)
        )
        async with self._sessions() as session:
            return [_to_hit(row, is_similarity=True) for row in await session.execute(stmt)]
