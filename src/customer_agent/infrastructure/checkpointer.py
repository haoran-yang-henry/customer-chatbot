"""LangGraph's Postgres checkpointer: the transcript and state of served sessions (phase 3).

LangGraph creates and versions its checkpoint tables itself (``setup``, its own migrations
table), next to Alembic's schema. The connection pool opens on ``start``, not on construction,
so a process that never serves a session (the evaluation) never connects. ``start`` does not
wait for the database either: the process starts and is live without one, as before; the
tables are created as soon as the database answers, and readiness waits for them (``ready``).
"""

import asyncio
from collections.abc import Sequence

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool
from sqlalchemy.engine import make_url

from customer_agent.observability import get_logger

log = get_logger("checkpoints")

SETUP_RETRY_SECONDS = 2.0


class PostgresCheckpointer:
    def __init__(self, database_url: str, types: Sequence[tuple[str, str]]) -> None:
        """``types``: the only classes a checkpoint may deserialize (module, name)."""
        # SQLAlchemy's URL names the driver (postgresql+psycopg); libpq does not know it.
        url = make_url(database_url).set(drivername="postgresql")
        self._pool = AsyncConnectionPool[AsyncConnection[DictRow]](
            url.render_as_string(hide_password=False),
            open=False,
            min_size=1,
            max_size=10,
            # What AsyncPostgresSaver expects of its connections.
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        )
        serde = JsonPlusSerializer(allowed_msgpack_modules=list(types))
        self.saver = AsyncPostgresSaver(self._pool, serde=serde)
        self._setup: asyncio.Task[None] | None = None

    @property
    def ready(self) -> bool:
        """The checkpoint tables exist: sessions can be served."""
        return self._setup is not None and self._setup.done() and not self._setup.cancelled()

    async def start(self) -> None:
        await self._pool.open(wait=False)
        self._setup = asyncio.create_task(self._set_up())

    async def _set_up(self) -> None:
        attempt = 0
        while True:
            try:
                await self.saver.setup()
            except Exception as exc:  # the database is not there yet: try again
                if attempt == 0:
                    log.warning("checkpoint tables not ready error=%s", type(exc).__name__)
                attempt += 1
                await asyncio.sleep(SETUP_RETRY_SECONDS)
            else:
                log.info("checkpoint tables ready attempts=%d", attempt + 1)
                return

    async def wait_ready(self) -> None:
        """For callers that need the tables now (tests, scripts)."""
        if self._setup is None:
            raise RuntimeError("start() the checkpointer first")
        await asyncio.shield(self._setup)

    async def aclose(self) -> None:
        if self._setup is not None:
            self._setup.cancel()
        await self._pool.close()
