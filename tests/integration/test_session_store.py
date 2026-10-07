"""PostgresSessionStore against a real database (migration 0002)."""

import asyncio

from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from customer_agent.application.ports import ConcurrentTurnError
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

    await sessions.save_turn(
        created.session_id, expected_version=0, state=WAITING, user="Wo ist sie?", reply=REPLY
    )
    loaded = await sessions.load(created.session_id)

    assert loaded is not None
    assert loaded.customer_id == "C-1002" and loaded.state == WAITING
    assert loaded.version == 1
    assert [(m.role, m.content) for m in loaded.messages] == [
        (Role.USER, "Wo ist sie?"),
        (Role.ASSISTANT, "Um welche Bestellung geht es?"),
    ]


async def test_unknown_session_loads_as_none(engine: AsyncEngine) -> None:
    assert await store(engine).load("0" * 32) is None


async def test_concurrent_turns_cannot_overwrite_each_other(engine: AsyncEngine) -> None:
    """Five turns started from the same version: exactly one wins, the rest are rejected."""
    sessions = store(engine)
    created = await sessions.create(tenant_id="nordelektro", customer_id=None)

    outcomes = await asyncio.gather(
        *(
            sessions.save_turn(
                created.session_id,
                expected_version=0,
                state=SessionState(),
                user=f"q{i}",
                reply=REPLY,
            )
            for i in range(5)
        ),
        return_exceptions=True,
    )
    loaded = await sessions.load(created.session_id)

    assert sum(o is None for o in outcomes) == 1
    assert all(isinstance(o, ConcurrentTurnError) for o in outcomes if o is not None)
    assert loaded is not None and loaded.version == 1
    assert [m.role for m in loaded.messages] == [Role.USER, Role.ASSISTANT]


async def test_sequential_turns_increment_the_version(engine: AsyncEngine) -> None:
    sessions = store(engine)
    created = await sessions.create(tenant_id="nordelektro", customer_id=None)

    for version in range(3):
        await sessions.save_turn(
            created.session_id,
            expected_version=version,
            state=SessionState(),
            user=f"q{version}",
            reply=REPLY,
        )
    loaded = await sessions.load(created.session_id)

    assert loaded is not None and loaded.version == 3 and len(loaded.messages) == 6
    assert [m.content for m in loaded.messages][::2] == ["q0", "q1", "q2"]
