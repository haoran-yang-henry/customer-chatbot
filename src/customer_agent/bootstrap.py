"""Composition root: builds adapters and use cases from settings.

Nothing else in the codebase reads settings to construct infrastructure. The API (lifespan),
the CLIs and the evaluation all obtain their objects from here.
"""

import asyncio
from dataclasses import dataclass

from openai import AsyncOpenAI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from customer_agent.application.answering import AnswerService
from customer_agent.application.chat import ChatService
from customer_agent.application.guardrails.classifier import PromptAttackClassifier
from customer_agent.application.orchestrator import ConversationOrchestrator
from customer_agent.application.ports import (
    AuditLog,
    Clock,
    EmbeddingProvider,
    LLMProvider,
    TicketService,
    TokenVerifier,
)
from customer_agent.application.retrieval import RetrievalService
from customer_agent.application.routing import IntentRouter
from customer_agent.config import Settings
from customer_agent.infrastructure.auth import JwtVerifier, jwks_resolver
from customer_agent.infrastructure.cassette import CassetteEmbeddings, CassetteLLM, open_cassette
from customer_agent.infrastructure.clock import SystemClock
from customer_agent.infrastructure.db.knowledge_repository import PostgresKnowledgeRepository
from customer_agent.infrastructure.db.models import EMBEDDING_DIMENSIONS
from customer_agent.infrastructure.db.order_repository import (
    PostgresAuditLog,
    PostgresOrderRepository,
    PostgresTicketService,
)
from customer_agent.infrastructure.db.session_repository import PostgresSessionStore
from customer_agent.infrastructure.embeddings import OpenAIEmbeddingProvider
from customer_agent.infrastructure.lifecycle import aclose_all
from customer_agent.infrastructure.llm import OpenAICompatibleLLM
from customer_agent.infrastructure.llm_cache import CachingLLM
from customer_agent.infrastructure.memory import InMemoryAuditLog, InMemoryTicketService
from customer_agent.infrastructure.prompts import load_prompt
from customer_agent.infrastructure.resilience import (
    BreakerEmbeddings,
    BreakerLLM,
    CircuitBreaker,
)
from customer_agent.observability import get_logger

log = get_logger("bootstrap")


class ConfigurationError(RuntimeError):
    """The configuration cannot work; the message tells the operator what to fix."""


def create_engine(settings: Settings) -> AsyncEngine:
    return create_async_engine(
        settings.database_url,
        pool_pre_ping=True,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        echo=settings.sql_echo,
    )


def create_knowledge_repository(engine: AsyncEngine) -> PostgresKnowledgeRepository:
    return PostgresKnowledgeRepository(async_sessionmaker(engine, expire_on_commit=False))


def check_embedding_dimensions(provider: EmbeddingProvider) -> None:
    if provider.dimensions != EMBEDDING_DIMENSIONS:
        raise ConfigurationError(
            f"embedding provider returns {provider.dimensions} dimensions, but the database "
            f"column has {EMBEDDING_DIMENSIONS}; change CA_EMBEDDING_DIMENSIONS or add a migration"
        )


def create_breaker(settings: Settings, dependency: str) -> CircuitBreaker:
    return CircuitBreaker(
        dependency,
        failure_threshold=settings.breaker_failure_threshold,
        reset_after=settings.breaker_reset_seconds,
    )


def create_embedding_provider(
    settings: Settings, breaker: CircuitBreaker | None = None
) -> EmbeddingProvider:
    if settings.eval_cassette is not None:
        cassette = open_cassette(settings.eval_cassette, settings.eval_cassette_record)
        inner = _embedding_provider(settings, breaker) if cassette.record else None
        return CassetteEmbeddings(
            cassette, inner, settings.embedding_model, settings.embedding_dimensions
        )
    return _embedding_provider(settings, breaker)


def _embedding_provider(settings: Settings, breaker: CircuitBreaker | None) -> EmbeddingProvider:
    if settings.openai_api_key is None:
        raise ConfigurationError("CA_OPENAI_API_KEY is not set (see .env.example)")
    client = AsyncOpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
        base_url=settings.openai_base_url,
        timeout=settings.embedding_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )
    provider = OpenAIEmbeddingProvider(
        client,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        query_timeout=settings.embedding_query_timeout_seconds,
    )
    check_embedding_dimensions(provider)
    return BreakerEmbeddings(provider, breaker or create_breaker(settings, "embeddings"))


def create_llm_providers(
    settings: Settings, breaker: CircuitBreaker | None = None
) -> dict[str, LLMProvider]:
    """One provider per role (``router``, ``answer``) with its own timeout, sharing one HTTP
    connection pool and one circuit breaker (same endpoint)."""
    if settings.eval_cassette is not None:
        cassette = open_cassette(settings.eval_cassette, settings.eval_cassette_record)
        inner = _llm_providers(settings, breaker) if cassette.record else None
        models = {
            "router": settings.router_llm_model or settings.llm_model,
            "answer": settings.answer_llm_model or settings.llm_model,
        }
        return {
            role: CassetteLLM(cassette, inner[role] if inner else None, model)
            for role, model in models.items()
        }
    return _llm_providers(settings, breaker)


def _llm_providers(settings: Settings, breaker: CircuitBreaker | None) -> dict[str, LLMProvider]:
    if settings.llm_api_key is None:
        raise ConfigurationError("CA_LLM_API_KEY is not set (see .env.example)")
    client = AsyncOpenAI(
        api_key=settings.llm_api_key.get_secret_value(),
        base_url=settings.llm_base_url,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        default_headers={"User-Agent": settings.llm_user_agent},
    )
    breaker = breaker or create_breaker(settings, "llm")
    roles = {
        "router": (
            settings.router_llm_model,
            settings.router_reasoning_effort,
            settings.router_timeout_seconds,
        ),
        "answer": (
            settings.answer_llm_model,
            settings.answer_reasoning_effort,
            settings.llm_timeout_seconds,
        ),
    }
    providers: dict[str, LLMProvider] = {}
    for role, (model_override, effort_override, timeout) in roles.items():
        model = model_override or settings.llm_model
        effort = effort_override or settings.llm_reasoning_effort
        llm: LLMProvider = BreakerLLM(
            OpenAICompatibleLLM(
                client.with_options(timeout=timeout),  # shares the connection pool
                model,
                session_header=settings.llm_session_header,
                reasoning_effort=effort,
            ),
            breaker,
        )
        if settings.llm_cache_dir is not None:
            namespace = f"{settings.llm_base_url or 'openai-default'}|{model}|reasoning={effort}"
            llm = CachingLLM(llm, settings.llm_cache_dir, namespace=namespace)
        providers[role] = llm
    return providers


def create_llm_provider(settings: Settings) -> LLMProvider:
    """The answer-role provider (also used by the evaluation judge)."""
    return create_llm_providers(settings)["answer"]


def create_token_verifier(settings: Settings) -> TokenVerifier | None:
    if settings.auth_issuer is None:
        return None
    issuer = settings.auth_issuer.rstrip("/")
    jwks_url = settings.auth_jwks_url or f"{issuer}/protocol/openid-connect/certs"
    return JwtVerifier(
        issuer=issuer, audience=settings.auth_audience, resolve_key=jwks_resolver(jwks_url)
    )


@dataclass
class Container:
    """Long-lived objects of one process; ``aclose`` releases connections."""

    engine: AsyncEngine
    agent: ConversationOrchestrator
    chat: ChatService
    llms: dict[str, LLMProvider]
    embeddings: EmbeddingProvider
    verifier: TokenVerifier | None
    prompts: dict[str, str]
    breakers: dict[str, CircuitBreaker]

    @property
    def models(self) -> dict[str, str]:
        return {role: llm.model_name for role, llm in self.llms.items()}

    async def readiness(self) -> dict[str, str]:
        """The database decides readiness. Breaker states are reported, but an open breaker
        keeps the instance ready: it still answers with the outage fallback (NFR-11), and
        taking every instance out of the load balancer would turn an outage into downtime."""
        checks: dict[str, str] = {name: breaker.state for name, breaker in self.breakers.items()}
        try:
            async with asyncio.timeout(2.0), self.engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
            checks["database"] = "ok"
        except Exception as exc:  # any failure means "not ready"
            # No traceback: probes run every few seconds, a database outage would flood the log.
            log.warning("readiness database=unavailable error=%s", type(exc).__name__)
            checks["database"] = "unavailable"
        return checks

    async def aclose(self) -> None:
        """Graceful shutdown: let in-flight turns finish, then release connections."""
        await self.chat.drain()
        # Both roles share one HTTP client; closing it once is enough, twice is harmless.
        await aclose_all(*self.llms.values(), self.embeddings)
        await self.engine.dispose()


def build_container(
    settings: Settings, clock: Clock | None = None, *, side_effects: bool = True
) -> Container:
    """``side_effects=False`` (evaluation) keeps tickets and audit entries in memory and lifts
    the ticket limit: evaluation cases are independent conversations, while the limit counts a
    customer's tickets across conversations (covered by unit tests instead)."""
    engine = create_engine(settings)
    clock = clock or SystemClock()
    breakers = {name: create_breaker(settings, name) for name in ("llm", "embeddings")}
    embeddings = create_embedding_provider(settings, breakers["embeddings"])
    llms = create_llm_providers(settings, breakers["llm"])
    answer_prompt = load_prompt("answer", settings.answer_prompt_version)
    router_prompt = load_prompt("router", settings.router_prompt_version)
    retrieval = RetrievalService(
        create_knowledge_repository(engine),
        embeddings,
        fulltext_weight=settings.retrieval_fulltext_weight,
    )
    answers = AnswerService(
        retrieval,
        llms["answer"],
        answer_prompt,
        clock,
        tenant_id=settings.tenant_id,
        min_similarity=settings.min_answer_similarity,
        top_k=settings.answer_top_k,
        max_tokens=settings.answer_max_tokens,
    )
    classifier = (
        PromptAttackClassifier(
            llms["router"],
            load_prompt("guard", settings.guard_prompt_version),
            threshold=settings.guard_classifier_threshold,
        )
        if settings.guard_classifier_enabled
        else None
    )
    db = async_sessionmaker(engine, expire_on_commit=False)
    tickets: TicketService = PostgresTicketService(db) if side_effects else InMemoryTicketService()
    audit: AuditLog = PostgresAuditLog(db) if side_effects else InMemoryAuditLog()
    agent = ConversationOrchestrator(
        IntentRouter(llms["router"], router_prompt),
        answers,
        orders=PostgresOrderRepository(db),
        tickets=tickets,
        audit=audit,
        clock=clock,
        tenant_id=settings.tenant_id,
        classifier=classifier,
        max_tickets_per_day=settings.max_tickets_per_day if side_effects else None,
    )
    sessions = PostgresSessionStore(db)
    prompts = {"answer": answer_prompt.ref, "router": router_prompt.ref}
    if classifier is not None:
        prompts["guard"] = classifier.prompt_ref
    return Container(
        engine=engine,
        agent=agent,
        chat=ChatService(agent, sessions, tenant_id=settings.tenant_id),
        llms=llms,
        embeddings=embeddings,
        verifier=create_token_verifier(settings),
        prompts=prompts,
        breakers=breakers,
    )
