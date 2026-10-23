"""PostgresSessionStore against a real database: ownership and the turn lease (migration 0005)."""

import asyncio

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

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
