"""PostgreSQL implementation of the ``SessionStore`` port."""

import uuid

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from customer_agent.application.ports import StoredSession
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
            )

    async def save_turn(
        self, session_id: str, *, state: SessionState, user: str, reply: AgentResponse
    ) -> None:
        async with self._sessions.begin() as session:
            # Row lock: concurrent requests on one session append in a well-defined order.
            row = await session.get(ChatSessionRow, session_id, with_for_update=True)
            if row is None:
                raise LookupError(session_id)
            position = await session.scalar(
                select(func.coalesce(func.max(ChatMessageRow.position), -1)).where(
                    ChatMessageRow.session_id == session_id
                )
            )
            next_position = (position if position is not None else -1) + 1
            session.add_all(
                [
                    ChatMessageRow(
                        session_id=session_id,
                        position=next_position,
                        role=Role.USER.value,
                        content=user,
                    ),
                    ChatMessageRow(
                        session_id=session_id,
                        position=next_position + 1,
                        role=Role.ASSISTANT.value,
                        content=reply.answer,
                        intent=reply.intent.value if reply.intent else None,
                        behavior=reply.behavior.value,
                        sources=list(reply.sources),
                    ),
                ]
            )
            await session.execute(
                update(ChatSessionRow)
                .where(ChatSessionRow.id == session_id)
                .values(
                    active_intent=state.active_intent.value if state.active_intent else None,
                    workflow_state=state.workflow.value,
                    slots=state.slots,
                    updated_at=func.now(),
                )
            )
