"""Composition root: builds adapters and use cases from settings.

Nothing else in the codebase reads settings to construct infrastructure. The API (lifespan),
the CLIs and the evaluation all obtain their objects from here.
"""

import asyncio
from dataclasses import dataclass
from typing import Literal

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from opentelemetry.sdk.trace import TracerProvider
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from customer_agent.application.answering import AnswerService
from customer_agent.application.chat import ChatService
from customer_agent.application.feedback import FeedbackPolicy
from customer_agent.application.guardrails.classifier import PromptAttackClassifier
from customer_agent.application.orchestrator import ConversationOrchestrator
from customer_agent.application.ports import (
    AuditLog,
    Clock,
    TicketService,
    TokenVerifier,
)
from customer_agent.application.retrieval import HybridRetriever
from customer_agent.application.turn_graph import CHECKPOINT_TYPES
from customer_agent.application.understanding import UnderstandAgent
from customer_agent.config import Settings
from customer_agent.infrastructure.auth import JwtVerifier, jwks_resolver
from customer_agent.infrastructure.cassette import CassetteEmbeddings, open_cassette
from customer_agent.infrastructure.checkpointer import PostgresCheckpointer
from customer_agent.infrastructure.clock import SystemClock
from customer_agent.infrastructure.db.knowledge_repository import PostgresKnowledgeRepository
from customer_agent.infrastructure.db.models import EMBEDDING_DIMENSIONS
from customer_agent.infrastructure.db.order_repository import (
    PostgresAuditLog,
    PostgresOrderRepository,
    PostgresTicketService,
)
from customer_agent.infrastructure.db.session_repository import PostgresSessionStore
from customer_agent.infrastructure.embeddings import ProviderEmbeddings
from customer_agent.infrastructure.lifecycle import aclose_all
from customer_agent.infrastructure.memory import InMemoryAuditLog, InMemoryTicketService
from customer_agent.infrastructure.models import BreakerChatModel, ProviderChatModel, model_label
from customer_agent.infrastructure.prompts import load_prompt
from customer_agent.infrastructure.recorded import CachedChatModel, ReplayChatModel
from customer_agent.infrastructure.resilience import BreakerEmbeddings, CircuitBreaker
from customer_agent.observability import configure_langfuse, get_logger
from customer_agent.observability.langfuse_prompts import prefetch_prompts

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


def check_embedding_dimensions(dimensions: int) -> None:
    if dimensions != EMBEDDING_DIMENSIONS:
        raise ConfigurationError(
            f"embedding provider returns {dimensions} dimensions, but the database "
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
) -> Embeddings:
    check_embedding_dimensions(settings.embedding_dimensions)
    if settings.eval_cassette is not None:
        cassette = open_cassette(settings.eval_cassette, settings.eval_cassette_record)
        inner = _embedding_provider(settings, breaker) if cassette.record else None
        return CassetteEmbeddings(
            cassette, inner, settings.embedding_model, settings.embedding_dimensions
        )
    return _embedding_provider(settings, breaker)


def _embedding_provider(settings: Settings, breaker: CircuitBreaker | None) -> Embeddings:
    if settings.openai_api_key is None:
        raise ConfigurationError("CA_OPENAI_API_KEY is not set (see .env.example)")
    provider = ProviderEmbeddings(
        openai_api_key=settings.openai_api_key,
        openai_api_base=settings.openai_base_url,
        request_timeout=settings.embedding_timeout_seconds,
        max_retries=settings.llm_max_retries,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        query_timeout=settings.embedding_query_timeout_seconds,
    )
    return BreakerEmbeddings(provider, breaker or create_breaker(settings, "embeddings"))


def create_chat_models(
    settings: Settings,
    breaker: CircuitBreaker | None = None,
    *,
    roles: tuple[Literal["router", "answer"], ...] = ("router", "answer"),
) -> dict[str, BaseChatModel]:
    """One chat model per role (``router``, ``answer``) with its own timeout, sharing one
    circuit breaker (same endpoint); replayed from the cassette in evaluation runs."""
    if settings.eval_cassette is not None:
        cassette = open_cassette(settings.eval_cassette, settings.eval_cassette_record)
        inner = _chat_models(settings, breaker, roles) if cassette.record else None
        labels = {
            "router": settings.router_llm_model or settings.llm_model,
            "answer": settings.answer_llm_model or settings.llm_model,
        }
        return {
            role: ReplayChatModel(
                name=role,
                cassette=cassette,
                inner=inner[role] if inner else None,
                label=labels[role],
            )
            for role in roles
        }
    return _chat_models(settings, breaker, roles)


def _chat_models(
    settings: Settings,
    breaker: CircuitBreaker | None,
    selected_roles: tuple[Literal["router", "answer"], ...],
) -> dict[str, BaseChatModel]:
    if settings.llm_api_key is None:
        raise ConfigurationError("CA_LLM_API_KEY is not set (see .env.example)")
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
    models: dict[str, BaseChatModel] = {}
    for role in selected_roles:
        model_override, effort_override, timeout = roles[role]
        name = model_override or settings.llm_model
        effort = effort_override or settings.llm_reasoning_effort
        model: BaseChatModel = BreakerChatModel(
            name=role,  # the run name in traces
            inner=ProviderChatModel(
                model_name=name,
                openai_api_key=settings.llm_api_key,
                openai_api_base=settings.llm_base_url,
                request_timeout=timeout,
                max_retries=settings.llm_max_retries,
                reasoning_effort=effort,
                default_headers={"User-Agent": settings.llm_user_agent},
                session_header=settings.llm_session_header,
                stream_usage=True,
            ),
            breaker=breaker,
        )
        if settings.llm_cache_dir is not None:
            namespace = f"{settings.llm_base_url or 'openai-default'}|{name}|reasoning={effort}"
            model = CachedChatModel(
                name=role, inner=model, directory=settings.llm_cache_dir, namespace=namespace
            )
        models[role] = model
    return models


def create_chat_model(settings: Settings) -> BaseChatModel:
    """The answer-role model (also used by the evaluation judge)."""
    return create_chat_models(settings, roles=("answer",))["answer"]


def start_langfuse(
    settings: Settings,
    tracer_provider: TracerProvider,
    version: str,
    environment: str | None = None,
) -> bool:
    """Langfuse tracing of turns, agent runs and model calls, when configured (see
    ``langfuse_enabled``). ``environment`` separates evaluation runs from chat traces (default:
    ``CA_APP_ENV``)."""
    key, secret = settings.langfuse_public_key, settings.langfuse_secret_key
    if not settings.langfuse_enabled or key is None or secret is None:
        return False
    client = configure_langfuse(
        public_key=key,
        secret_key=secret.get_secret_value(),
        base_url=settings.langfuse_base_url,
        environment=environment or settings.app_env,
        release=version,
        tracer_provider=tracer_provider,
    )
    prefetch_prompts(client, [load_prompt(name, v) for name, v in prompts_in_use(settings)])
    log.info("langfuse tracing enabled base_url=%s", settings.langfuse_base_url or "default")
    return True


def prompts_in_use(settings: Settings) -> list[tuple[str, int]]:
    """The prompt versions this configuration renders (Langfuse label ``production``)."""
    return [
        ("router", settings.router_prompt_version),
        ("answer", settings.answer_prompt_version),
        ("guard", settings.guard_prompt_version),
        ("judge", settings.judge_prompt_version),
    ]


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
    checkpointer: PostgresCheckpointer
    agent: ConversationOrchestrator
    chat: ChatService
    llms: dict[str, BaseChatModel]
    embeddings: Embeddings
    verifier: TokenVerifier | None
    prompts: dict[str, str]
    breakers: dict[str, CircuitBreaker]

    @property
    def models(self) -> dict[str, str]:
        return {role: model_label(llm) for role, llm in self.llms.items()}

    async def readiness(self) -> dict[str, str]:
        """The database decides readiness. Breaker states are reported, but an open breaker
        keeps the instance ready: it still answers with the outage fallback (NFR-11), and
        taking every instance out of the load balancer would turn an outage into downtime."""
        checks: dict[str, str] = {name: breaker.state for name, breaker in self.breakers.items()}
        checks["checkpoints"] = "ok" if self.checkpointer.ready else "pending"
        try:
            async with asyncio.timeout(2.0), self.engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
            checks["database"] = "ok"
        except Exception as exc:  # any failure means "not ready"
            # No traceback: probes run every few seconds, a database outage would flood the log.
            log.warning("readiness database=unavailable error=%s", type(exc).__name__)
            checks["database"] = "unavailable"
        return checks

    async def start(self) -> None:
        """Before serving sessions: connect the checkpointer and create its tables if missing,
        in the background (readiness waits for them)."""
        await self.checkpointer.start()

    async def aclose(self) -> None:
        """Graceful shutdown: let in-flight turns finish, then release connections."""
        await self.chat.drain()
        await aclose_all(*self.llms.values(), self.embeddings)
        await self.checkpointer.aclose()
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
    llms = create_chat_models(settings, breakers["llm"])
    answer_prompt = load_prompt("answer", settings.answer_prompt_version)
    router_prompt = load_prompt("router", settings.router_prompt_version)
    retriever = HybridRetriever(
        search=create_knowledge_repository(engine),
        embeddings=embeddings,
        clock=clock,
        tenant_id=settings.tenant_id,
        k=settings.answer_top_k,
        fulltext_weight=settings.retrieval_fulltext_weight,
    )
    answers = AnswerService(
        retriever,
        llms["answer"],
        answer_prompt,
        min_similarity=settings.min_answer_similarity,
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
    checkpointer = PostgresCheckpointer(settings.database_url, CHECKPOINT_TYPES)
    agent = ConversationOrchestrator(
        UnderstandAgent(llms["router"], router_prompt, classifier=classifier),
        answers,
        orders=PostgresOrderRepository(db),
        tickets=tickets,
        audit=audit,
        clock=clock,
        tenant_id=settings.tenant_id,
        checkpointer=checkpointer.saver,
        max_tickets_per_day=settings.max_tickets_per_day if side_effects else None,
        feedback=FeedbackPolicy(settings.feedback_sample_rate, settings.feedback_min_rounds),
    )
    sessions = PostgresSessionStore(db)
    prompts = {"answer": answer_prompt.ref, "router": router_prompt.ref}
    if classifier is not None:
        prompts["guard"] = classifier.prompt_ref
    return Container(
        engine=engine,
        checkpointer=checkpointer,
        agent=agent,
        chat=ChatService(agent, sessions, tenant_id=settings.tenant_id),
        llms=llms,
        embeddings=embeddings,
        verifier=create_token_verifier(settings),
        prompts=prompts,
        breakers=breakers,
    )
