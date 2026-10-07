"""PostgreSQL implementation of the ``SessionStore`` port."""

import uuid

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from customer_agent.application.ports import ConcurrentTurnError, StoredSession
from customer_agent.domain.conversation import AgentResponse, Message, Role
from customer_agent.domain.intents import Intent
from customer_agent.domain.session import SessionState, WorkflowState
from customer_agent.infrastructure.db.models import ChatMessageRow, ChatSessionRow


def _state(row: ChatSessionRow) -> SessionState:
    return SessionState(
        active_intent=Intent(row.active_intent) if row.active_intent else None,
        workflow=WorkflowState(row.workflow_state),
        slots=dict(row.slots),
    )


class PostgresSessionStore:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def create(self, *, tenant_id: str, customer_id: str | None) -> StoredSession:
        row = ChatSessionRow(
            id=uuid.uuid4().hex,
            tenant_id=tenant_id,
            customer_id=customer_id,
            active_intent=None,
            workflow_state=WorkflowState.IDLE.value,
            slots={},
            version=0,
        )
        async with self._sessions.begin() as session:
            session.add(row)
        return StoredSession(
            session_id=row.id, customer_id=customer_id, state=SessionState(), messages=[]
        )

    async def load(self, session_id: str) -> StoredSession | None:
        async with self._sessions() as session:
            row = await session.get(ChatSessionRow, session_id)
            if row is None:
                return None
            messages = await session.scalars(
                select(ChatMessageRow)
                .where(ChatMessageRow.session_id == session_id)
                .order_by(ChatMessageRow.position)
            )
            return StoredSession(
                session_id=row.id,
                customer_id=row.customer_id,
                state=_state(row),
                messages=[Message(role=Role(m.role), content=m.content) for m in messages],
                version=row.version,
            )

    async def save_turn(
        self,
        session_id: str,
        *,
        expected_version: int,
        state: SessionState,
        user: str,
        reply: AgentResponse,
    ) -> None:
        async with self._sessions.begin() as session:
            # Optimistic concurrency: the UPDATE only matches the version this turn started from.
            # A concurrent turn blocks on the row lock, then sees the new version and matches 0
            # rows, so neither the state nor the transcript can be overwritten silently.
            updated = await session.execute(
                update(ChatSessionRow)
                .where(ChatSessionRow.id == session_id)
                .where(ChatSessionRow.version == expected_version)
                .values(
                    active_intent=state.active_intent.value if state.active_intent else None,
                    workflow_state=state.workflow.value,
                    slots=state.slots,
                    version=ChatSessionRow.version + 1,
                    updated_at=func.now(),
                )
                .returning(ChatSessionRow.id)
            )
            if updated.first() is None:
                if await session.get(ChatSessionRow, session_id) is None:
                    raise LookupError(session_id)
                raise ConcurrentTurnError(session_id)
            last = await session.scalar(
                select(func.max(ChatMessageRow.position)).where(
                    ChatMessageRow.session_id == session_id
                )
            )
            position = -1 if last is None else last
            session.add_all(
                [
                    ChatMessageRow(
                        session_id=session_id,
                        position=position + 1,
                        role=Role.USER.value,
                        content=user,
                    ),
                    ChatMessageRow(
                        session_id=session_id,
                        position=position + 2,
                        role=Role.ASSISTANT.value,
                        content=reply.answer,
                        intent=reply.intent.value if reply.intent else None,
                        behavior=reply.behavior.value,
                        sources=list(reply.sources),
                    ),
                ]
            )
