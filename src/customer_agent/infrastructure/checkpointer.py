"""LangGraph's Postgres checkpointer: the transcript and state of served sessions (phase 3).

LangGraph creates and versions its checkpoint tables itself (``setup``, its own migrations
table), next to Alembic's schema. The connection pool opens on ``start``, not on construction,
so a process that never serves a session (the evaluation) never connects.
"""

from collections.abc import Sequence

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool
from sqlalchemy.engine import make_url


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

    async def start(self) -> None:
        await self._pool.open()
        await self.saver.setup()

    async def aclose(self) -> None:
        await self._pool.close()
