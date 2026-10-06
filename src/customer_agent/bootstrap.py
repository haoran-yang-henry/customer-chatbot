"""Composition root for non-HTTP entry points (CLIs) and, from Day 3, the API.

Builds adapters from settings. Nothing else in the codebase reads settings to construct
infrastructure.
"""

from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from customer_agent.application.ports import EmbeddingProvider
from customer_agent.config import Settings
from customer_agent.infrastructure.db.knowledge_repository import PostgresKnowledgeRepository
from customer_agent.infrastructure.db.models import EMBEDDING_DIMENSIONS


class ConfigurationError(RuntimeError):
    """The configuration cannot work; the message tells the operator what to fix."""


def create_engine(settings: Settings) -> AsyncEngine:
    return create_async_engine(settings.database_url, pool_pre_ping=True, echo=settings.sql_echo)


def create_knowledge_repository(engine: AsyncEngine) -> PostgresKnowledgeRepository:
    return PostgresKnowledgeRepository(async_sessionmaker(engine, expire_on_commit=False))


def check_embedding_dimensions(provider: EmbeddingProvider) -> None:
    if provider.dimensions != EMBEDDING_DIMENSIONS:
        raise ConfigurationError(
            f"embedding provider returns {provider.dimensions} dimensions, but the database "
            f"column has {EMBEDDING_DIMENSIONS}; change CA_EMBEDDING_DIMENSIONS or add a migration"
        )


def create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    from openai import AsyncOpenAI

    from customer_agent.infrastructure.embeddings import OpenAIEmbeddingProvider

    if settings.openai_api_key is None:
        raise ConfigurationError("CA_OPENAI_API_KEY is not set (see .env.example)")
    client = AsyncOpenAI(
        api_key=settings.openai_api_key.get_secret_value(), base_url=settings.openai_base_url
    )
    provider = OpenAIEmbeddingProvider(
        client, model=settings.embedding_model, dimensions=settings.embedding_dimensions
    )
    check_embedding_dimensions(provider)
    return provider
