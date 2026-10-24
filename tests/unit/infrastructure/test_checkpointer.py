"""The checkpointer starts without a database (rebuild phase 3): liveness must not depend on it."""

import asyncio
import time

from customer_agent.application.turn_graph import CHECKPOINT_TYPES
from customer_agent.infrastructure.checkpointer import PostgresCheckpointer

NOWHERE = "postgresql+psycopg://agent:agent@127.0.0.1:9/none"  # nothing listens here


async def test_start_does_not_wait_for_the_database_and_reports_not_ready() -> None:
    checkpointer = PostgresCheckpointer(NOWHERE, CHECKPOINT_TYPES)
    started = time.perf_counter()

    await checkpointer.start()
    await asyncio.sleep(0.1)  # the setup task has tried at least once

    assert time.perf_counter() - started < 1.0
    assert not checkpointer.ready
    await checkpointer.aclose()
