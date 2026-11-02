"""Daily message count of the public demo (migration 0007)."""

from datetime import date

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from customer_agent.infrastructure.db.models import DailyUsageRow


class PostgresUsageCounter:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def count_message(self, day: date) -> int:
        # One upsert counts and reads: concurrent requests each get their own number.
        statement = (
            insert(DailyUsageRow)
            .values(day=day, messages=1)
            .on_conflict_do_update(
                index_elements=[DailyUsageRow.day],
                set_={"messages": DailyUsageRow.messages + 1},
            )
            .returning(DailyUsageRow.messages)
        )
        async with self._sessions.begin() as session:
            return (await session.execute(statement)).scalar_one()
