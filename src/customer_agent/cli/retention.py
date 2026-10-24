"""``customer-agent-retention [--days 90]``: delete conversations past the retention period.

policy-privacy keeps conversations for 90 days (``CA_RETENTION_DAYS``) after their last turn. A
session's masked transcript and state are its checkpoint thread: the thread goes first, then the
session row and its feedback. Each session is deleted on its own, so an interrupted run leaves
whole sessions behind, and the next run finishes them. Run it daily (cron, Kubernetes CronJob).
"""

import argparse
import asyncio
from datetime import UTC, datetime, timedelta

from sqlalchemy.ext.asyncio import async_sessionmaker

from customer_agent.application.turn_graph import CHECKPOINT_TYPES
from customer_agent.bootstrap import create_engine
from customer_agent.config import get_settings
from customer_agent.infrastructure.checkpointer import PostgresCheckpointer
from customer_agent.infrastructure.db.session_repository import PostgresSessionStore


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, help="Retention period (default: CA_RETENTION_DAYS).")
    return parser.parse_args(argv)


async def purge(
    sessions: PostgresSessionStore, checkpointer: PostgresCheckpointer, before: datetime
) -> int:
    expired = await sessions.idle_since(before)
    for session_id in expired:
        await checkpointer.saver.adelete_thread(session_id)
        await sessions.delete(session_id)
    return len(expired)


async def _run(days: int) -> int:
    settings = get_settings()
    engine = create_engine(settings)
    checkpointer = PostgresCheckpointer(settings.database_url, CHECKPOINT_TYPES)
    try:
        await checkpointer.start()
        await checkpointer.wait_ready()
        sessions = PostgresSessionStore(async_sessionmaker(engine, expire_on_commit=False))
        return await purge(sessions, checkpointer, datetime.now(UTC) - timedelta(days=days))
    finally:
        await checkpointer.aclose()
        await engine.dispose()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    days = args.days or get_settings().retention_days
    deleted = asyncio.run(_run(days))
    print(f"deleted {deleted} conversation(s) without a turn for {days} days")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
