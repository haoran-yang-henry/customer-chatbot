"""PostgreSQL implementation of the ``SessionStore`` port: ownership, the turn lease and the
served turns with their ratings.

The transcript and the state are the turn graph's checkpoint (``infrastructure/checkpointer``).
"""

import uuid
from datetime import timedelta

from sqlalchemy import func, or_, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from customer_agent.application.ports import StoredSession, StoredTurn
from customer_agent.infrastructure.db.models import ChatSessionRow, ChatTurnRow

# Longer than any turn (model timeouts included): an older lease belongs to a crashed process.
LEASE_TIMEOUT = timedelta(minutes=2)


class PostgresSessionStore:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def create(self, *, tenant_id: str, customer_id: str | None) -> StoredSession:
        row = ChatSessionRow(id=uuid.uuid4().hex, tenant_id=tenant_id, customer_id=customer_id)
        async with self._sessions.begin() as session:
            session.add(row)
        return StoredSession(session_id=row.id, tenant_id=tenant_id, customer_id=customer_id)

    async def load(self, session_id: str) -> StoredSession | None:
        async with self._sessions() as session:
            row = await session.get(ChatSessionRow, session_id)
            if row is None:
                return None
            return StoredSession(
                session_id=row.id, tenant_id=row.tenant_id, customer_id=row.customer_id
            )

    async def acquire(self, session_id: str) -> bool:
        # One conditional UPDATE: of several concurrent turns exactly one matches; the others
        # wait for its row lock, then see the lease and match no row. No connection is held
        # while the turn runs.
        async with self._sessions.begin() as session:
            leased = await session.execute(
                update(ChatSessionRow)
                .where(ChatSessionRow.id == session_id)
                .where(
                    or_(
                        ChatSessionRow.turn_started_at.is_(None),
                        ChatSessionRow.turn_started_at < func.now() - LEASE_TIMEOUT,
                    )
                )
                .values(turn_started_at=func.now())
                .returning(ChatSessionRow.id)
            )
            return leased.first() is not None

    async def release(self, session_id: str) -> None:
        async with self._sessions.begin() as session:
            await session.execute(
                update(ChatSessionRow)
                .where(ChatSessionRow.id == session_id)
                .values(turn_started_at=None, updated_at=func.now())
            )

    async def record_turn(self, turn: StoredTurn) -> None:
        row = {"session_id": turn.session_id, "request_id": turn.request_id}
        async with self._sessions.begin() as session:
            # Request ids may come from the client (X-Request-ID): a repeat overwrites.
            await session.execute(
                insert(ChatTurnRow)
                .values(**row, trace_id=turn.trace_id)
                .on_conflict_do_update(
                    index_elements=[ChatTurnRow.session_id, ChatTurnRow.request_id],
                    set_={"trace_id": turn.trace_id, "helpful": None, "comment": None},
                )
            )

    async def rate_turn(
        self, session_id: str, request_id: str, *, helpful: bool, comment: str | None
    ) -> StoredTurn | None:
        async with self._sessions.begin() as session:
            rated = await session.execute(
                update(ChatTurnRow)
                .where(ChatTurnRow.session_id == session_id)
                .where(ChatTurnRow.request_id == request_id)
                .values(helpful=helpful, comment=comment, rated_at=func.now())
                .returning(ChatTurnRow.trace_id)
            )
            row = rated.first()
        if row is None:
            return None
        return StoredTurn(session_id=session_id, request_id=request_id, trace_id=row.trace_id)
