"""PostgresSessionStore against a real database: ownership and the turn lease (migration 0005)."""

import asyncio

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from customer_agent.application.ports import AskedFeedback
from customer_agent.infrastructure.db.session_repository import PostgresSessionStore


def store(engine: AsyncEngine) -> PostgresSessionStore:
    return PostgresSessionStore(async_sessionmaker(engine, expire_on_commit=False))


async def test_create_and_load_roundtrip(engine: AsyncEngine) -> None:
    sessions = store(engine)
    created = await sessions.create(tenant_id="nordelektro", customer_id="C-1002")

    loaded = await sessions.load(created.session_id)

    assert loaded == created


async def test_unknown_session_loads_as_none(engine: AsyncEngine) -> None:
    assert await store(engine).load("0" * 32) is None


async def test_of_concurrent_turns_exactly_one_gets_the_lease(engine: AsyncEngine) -> None:
    sessions = store(engine)
    created = await sessions.create(tenant_id="nordelektro", customer_id=None)

    leased = await asyncio.gather(*(sessions.acquire(created.session_id) for _ in range(5)))

    assert sorted(leased) == [False, False, False, False, True]


async def test_a_released_lease_can_be_taken_again(engine: AsyncEngine) -> None:
    sessions = store(engine)
    created = await sessions.create(tenant_id="nordelektro", customer_id=None)

    assert await sessions.acquire(created.session_id)
    await sessions.release(created.session_id)

    assert await sessions.acquire(created.session_id)


async def test_an_abandoned_lease_is_taken_over(engine: AsyncEngine) -> None:
    """A process that crashed mid-turn must not lock its session forever."""
    sessions = store(engine)
    created = await sessions.create(tenant_id="nordelektro", customer_id=None)
    assert await sessions.acquire(created.session_id)
    async with engine.begin() as connection:
        await connection.execute(
            text("UPDATE chat_sessions SET turn_started_at = now() - interval '3 minutes'")
        )

    assert await sessions.acquire(created.session_id)


async def test_an_unknown_session_gets_no_lease(engine: AsyncEngine) -> None:
    assert not await store(engine).acquire("0" * 32)


async def test_only_an_asked_session_is_rated(engine: AsyncEngine) -> None:
    sessions = store(engine)
    asked = await sessions.create(tenant_id="nordelektro", customer_id="C-1001")
    never = await sessions.create(tenant_id="nordelektro", customer_id="C-1001")
    question = AskedFeedback(session_id=asked.session_id, request_id="r3", trace_id="ab" * 16)
    await sessions.ask_feedback(question)

    rated = await sessions.rate_session(asked.session_id, helpful=False, comment="Falsch")
    elsewhere = await sessions.rate_session(never.session_id, helpful=True, comment=None)

    assert rated == question and elsewhere is None
    async with engine.connect() as connection:
        row = (await connection.execute(text("SELECT helpful, comment FROM chat_feedback"))).one()
    assert tuple(row) == (False, "Falsch")


async def test_a_session_is_asked_once(engine: AsyncEngine) -> None:
    sessions = store(engine)
    created = await sessions.create(tenant_id="nordelektro", customer_id=None)

    for request_id in ("r3", "r4"):
        await sessions.ask_feedback(
            AskedFeedback(session_id=created.session_id, request_id=request_id, trace_id=None)
        )

    async with engine.connect() as connection:
        rows = (await connection.execute(text("SELECT request_id FROM chat_feedback"))).all()
    assert [r[0] for r in rows] == ["r3"]
