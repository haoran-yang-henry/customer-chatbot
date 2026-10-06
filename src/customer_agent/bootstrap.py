"""Composition root: builds adapters and use cases from settings.

Nothing else in the codebase reads settings to construct infrastructure. The API (lifespan),
the CLIs and the evaluation all obtain their objects from here.
"""

from dataclasses import dataclass

from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from customer_agent.application.answering import AnswerService
from customer_agent.application.orchestrator import ConversationOrchestrator
from customer_agent.application.ports import Clock, EmbeddingProvider, LLMProvider
from customer_agent.application.retrieval import RetrievalService
from customer_agent.config import Settings
from customer_agent.infrastructure.clock import SystemClock
from customer_agent.infrastructure.db.knowledge_repository import PostgresKnowledgeRepository
from customer_agent.infrastructure.db.models import EMBEDDING_DIMENSIONS
from customer_agent.infrastructure.embeddings import OpenAIEmbeddingProvider
from customer_agent.infrastructure.llm import OpenAICompatibleLLM
from customer_agent.infrastructure.llm_cache import CachingLLM
from customer_agent.infrastructure.prompts import load_prompt


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


def create_llm_provider(settings: Settings) -> LLMProvider:
    if settings.llm_api_key is None:
        raise ConfigurationError("CA_LLM_API_KEY is not set (see .env.example)")
    client = AsyncOpenAI(
        api_key=settings.llm_api_key.get_secret_value(),
        base_url=settings.llm_base_url,
        timeout=settings.llm_timeout_seconds,
        max_retries=1,
        default_headers={"User-Agent": settings.llm_user_agent},
    )
    llm: LLMProvider = OpenAICompatibleLLM(
        client,
        settings.llm_model,
        session_header=settings.llm_session_header,
        reasoning_effort=settings.llm_reasoning_effort,
    )
    if settings.llm_cache_dir is not None:
        llm = CachingLLM(llm, settings.llm_cache_dir)
    return llm


@dataclass
class Container:
    """Long-lived objects of one process; ``aclose`` releases connections."""

    engine: AsyncEngine
    agent: ConversationOrchestrator
    llm_model: str
    answer_prompt: str

    async def aclose(self) -> None:
        await self.engine.dispose()


def build_container(settings: Settings, clock: Clock | None = None) -> Container:
    engine = create_engine(settings)
    embeddings = create_embedding_provider(settings)
    llm = create_llm_provider(settings)
    prompt = load_prompt("answer", settings.answer_prompt_version)
    retrieval = RetrievalService(create_knowledge_repository(engine), embeddings)
    answers = AnswerService(
        retrieval,
        llm,
        prompt,
        clock or SystemClock(),
        tenant_id=settings.tenant_id,
        min_similarity=settings.min_answer_similarity,
    )
    return Container(
        engine=engine,
        agent=ConversationOrchestrator(answers),
        llm_model=llm.model_name,
        answer_prompt=prompt.ref,
    )
