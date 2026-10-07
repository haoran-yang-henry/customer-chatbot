"""PostgresSessionStore against a real database (migration 0002)."""

import asyncio

from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from customer_agent.domain.conversation import AgentResponse, Behavior, Role
from customer_agent.domain.intents import Intent
from customer_agent.domain.session import SessionState, WorkflowState
from customer_agent.infrastructure.db.session_repository import PostgresSessionStore


def store(engine: AsyncEngine) -> PostgresSessionStore:
    return PostgresSessionStore(async_sessionmaker(engine, expire_on_commit=False))


REPLY = AgentResponse(
    answer="Um welche Bestellung geht es?", behavior=Behavior.CLARIFY, intent=Intent.ORDER_STATUS
)
WAITING = SessionState(active_intent=Intent.ORDER_STATUS, workflow=WorkflowState.AWAITING_ORDER_ID)


async def test_create_save_and_load_roundtrip(engine: AsyncEngine) -> None:
    sessions = store(engine)
    created = await sessions.create(tenant_id="nordelektro", customer_id="C-1002")

    await sessions.save_turn(created.session_id, state=WAITING, user="Wo ist sie?", reply=REPLY)
    loaded = await sessions.load(created.session_id)

    assert loaded is not None
    assert loaded.customer_id == "C-1002" and loaded.state == WAITING
    assert [(m.role, m.content) for m in loaded.messages] == [
        (Role.USER, "Wo ist sie?"),
        (Role.ASSISTANT, "Um welche Bestellung geht es?"),
    ]


async def test_unknown_session_loads_as_none(engine: AsyncEngine) -> None:
    assert await store(engine).load("0" * 32) is None


async def test_concurrent_turns_on_one_session_keep_a_consistent_order(engine: AsyncEngine) -> None:
    sessions = store(engine)
    created = await sessions.create(tenant_id="nordelektro", customer_id=None)

    await asyncio.gather(
        *(
            sessions.save_turn(created.session_id, state=SessionState(), user=f"q{i}", reply=REPLY)
            for i in range(5)
        )
    )
    loaded = await sessions.load(created.session_id)

    assert loaded is not None and len(loaded.messages) == 10
    roles = [m.role for m in loaded.messages]
    assert roles == [Role.USER, Role.ASSISTANT] * 5  # no interleaving thanks to the row lock
