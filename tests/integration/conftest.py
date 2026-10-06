"""Integration fixtures: a throwaway pgvector Postgres per test session, migrated with Alembic.

``--embeddings=openai`` swaps the deterministic hashing embedder for the real provider from
``.env``, so every integration test also runs under production semantics.
"""

from collections.abc import AsyncIterator, Iterator

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from testcontainers.community.postgres import PostgresContainer

from customer_agent.application.ports import EmbeddingProvider
from customer_agent.bootstrap import create_embedding_provider, create_knowledge_repository
from customer_agent.config import Settings
from customer_agent.infrastructure.db.knowledge_repository import PostgresKnowledgeRepository
from customer_agent.infrastructure.db.models import EMBEDDING_DIMENSIONS
from customer_agent.infrastructure.embeddings import HashingEmbeddingProvider
from tests.conftest import PROJECT_ROOT

POSTGRES_IMAGE = "pgvector/pgvector:pg16"  # same image as docker-compose.yml


@pytest.fixture(scope="session")
def database_url() -> Iterator[str]:
    with PostgresContainer(POSTGRES_IMAGE, driver="psycopg") as postgres:
        url = postgres.get_connection_url()
        config = Config(str(PROJECT_ROOT / "alembic.ini"))
        config.set_main_option("sqlalchemy.url", url)
        command.upgrade(config, "head")
        yield url


@pytest.fixture
async def engine(database_url: str) -> AsyncIterator[AsyncEngine]:
    engine = create_async_engine(database_url)
    async with engine.begin() as connection:
        await connection.execute(text("TRUNCATE knowledge_documents CASCADE"))
    yield engine
    await engine.dispose()


@pytest.fixture
def repository(engine: AsyncEngine) -> PostgresKnowledgeRepository:
    return create_knowledge_repository(engine)


@pytest.fixture
def embeddings(request: pytest.FixtureRequest) -> EmbeddingProvider:
    """Function-scoped: the OpenAI client's connection pool is bound to the test's event loop."""
    if request.config.getoption("--embeddings") == "openai":
        return create_embedding_provider(Settings())
    return HashingEmbeddingProvider(dimensions=EMBEDDING_DIMENSIONS)
