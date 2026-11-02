"""PostgresUsageCounter against a real database: the demo's daily message count (migration 0007)."""

import asyncio
from datetime import date

from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from customer_agent.infrastructure.db.usage_repository import PostgresUsageCounter

DAY = date(2026, 11, 2)


def counter(engine: AsyncEngine) -> PostgresUsageCounter:
    return PostgresUsageCounter(async_sessionmaker(engine, expire_on_commit=False))


async def test_messages_are_counted_per_day(engine: AsyncEngine) -> None:
    usage = counter(engine)

    counts = [await usage.count_message(DAY) for _ in range(3)]

    assert counts == [1, 2, 3]
    assert await usage.count_message(date(2026, 11, 3)) == 1  # a new day starts at one


async def test_concurrent_messages_each_get_their_own_count(engine: AsyncEngine) -> None:
    usage = counter(engine)

    counts = await asyncio.gather(*(usage.count_message(DAY) for _ in range(10)))

    assert sorted(counts) == list(range(1, 11))
