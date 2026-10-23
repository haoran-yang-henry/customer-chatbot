"""Test doubles for the application ports."""

from collections.abc import AsyncIterator
from datetime import date
from typing import TYPE_CHECKING, Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.retrievers import BaseRetriever
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import Field

from customer_agent.application.ports import (
    AskedFeedback,
    ProviderUnavailableError,
    StoredSession,
)
from customer_agent.domain.knowledge import SourceType, TrustLevel
from customer_agent.domain.orders import Order
from customer_agent.domain.retrieval import SearchMode, SearchScope
from customer_agent.infrastructure.memory import InMemoryAuditLog, InMemoryTicketService

if TYPE_CHECKING:
    from customer_agent.application.feedback import FeedbackPolicy
    from customer_agent.application.guardrails.classifier import PromptAttackClassifier
    from customer_agent.application.orchestrator import ConversationOrchestrator
    from customer_agent.application.retrieval import HybridRetriever


def document(
    doc_id: str,
    similarity: float | None,
    content: str = "Inhalt.",
    *,
    trust: TrustLevel = TrustLevel.OFFICIAL,
) -> Document:
    """A retrieved knowledge document as ``HybridRetriever`` returns it."""
    metadata = {
        "document_id": doc_id,
        "title": doc_id,
        "source_type": SourceType.POLICY.value,
        "trust_level": trust.value,
        "score": 0.03,
        "similarity": similarity,
        "chunk_ids": [hash(doc_id) % 10_000],
    }
    return Document(page_content=content, metadata=metadata)


class FakeRetriever(BaseRetriever):
    """Returns its documents for every query and records the queries."""

    documents: list[Document]
    queries: list[str] = Field(default_factory=list)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        self.queries.append(query)
        return self.documents


async def search(
    retriever: "HybridRetriever",
    query: str,
    scope: SearchScope,
    *,
    k: int = 5,
    mode: SearchMode = SearchMode.HYBRID,
) -> list[Document]:
    """One search with this scope, depth and mode; the retriever's own are left as they are."""
    configured = retriever.model_copy(
        update={
            "clock": FixedClock(scope.today),
            "tenant_id": scope.tenant_id,
            "k": k,
            "mode": mode,
        }
    )
    return await configured.ainvoke(query)


class FakeLLM(BaseChatModel):
    """Chat-model double: a scripted reply, streamed in small pieces to exercise the stream
    filter. Records every call's messages, keyword arguments (``max_tokens``, ``temperature``)
    and the ``session_id`` keyword the caller passed."""

    reply: str = ""
    fail: bool = False
    piece: int = 7
    usage: tuple[int, int] | None = None
    calls: list[list[BaseMessage]] = Field(default_factory=list)
    kwargs: list[dict[str, Any]] = Field(default_factory=list)
    session_ids: list[str | None] = Field(default_factory=list)

    def __init__(self, reply: str = "", **fields: Any) -> None:
        super().__init__(**{"reply": reply, **fields})

    @property
    def _llm_type(self) -> str:
        return "fake-llm"

    def _record(
        self,
        messages: list[BaseMessage],
        kwargs: dict[str, Any],
        run_manager: AsyncCallbackManagerForLLMRun | CallbackManagerForLLMRun | None,
    ) -> str:
        """The reply for this call; raises like a provider outage when ``fail`` is set."""
        self.calls.append(list(messages))
        self.kwargs.append(kwargs)
        self.session_ids.append(kwargs.get("session_id"))
        if self.fail:
            raise ProviderUnavailableError("fake outage")
        return self.reply

    def _usage(self) -> UsageMetadata | None:
        if self.usage is None:
            return None
        prompt, completion = self.usage
        return UsageMetadata(
            input_tokens=prompt, output_tokens=completion, total_tokens=prompt + completion
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        text = self._record(messages, kwargs, run_manager)
        message = AIMessage(content=text, usage_metadata=self._usage())
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        text = self._record(messages, kwargs, run_manager)
        message = AIMessage(content=text, usage_metadata=self._usage())
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        text = self._record(messages, kwargs, run_manager)
        for i in range(0, len(text), self.piece):
            yield ChatGenerationChunk(message=AIMessageChunk(content=text[i : i + self.piece]))


class FixedClock:
    def __init__(self, today: date) -> None:
        self._today = today

    def today(self) -> date:
        return self._today


class ScriptedRouterLLM(FakeLLM):
    """Answers router prompts with a JSON routing and answer prompts with ``answer``."""

    routing: str
    router_calls: int = 0
    router_prompts: list[list[BaseMessage]] = Field(default_factory=list)
    router_session_ids: list[str | None] = Field(default_factory=list)

    def __init__(
        self, routing: str, answer: str = "Ok. [policy-returns-v5]", **fields: Any
    ) -> None:
        super().__init__(answer, routing=routing, **fields)

    def _record(
        self,
        messages: list[BaseMessage],
        kwargs: dict[str, Any],
        run_manager: AsyncCallbackManagerForLLMRun | CallbackManagerForLLMRun | None,
    ) -> str:
        if "You classify customer messages" not in messages[0].text:
            return super()._record(messages, kwargs, run_manager)
        self.router_calls += 1
        self.router_prompts.append(list(messages))
        self.router_session_ids.append(kwargs.get("session_id"))
        if self.fail:
            raise ProviderUnavailableError("fake outage")
        return self.routing


class InMemorySessionStore:
    def __init__(self) -> None:
        self.sessions: dict[str, StoredSession] = {}
        self.leased: set[str] = set()
        self.asked: dict[str, AskedFeedback] = {}
        self.ratings: dict[str, tuple[bool, str | None]] = {}

    async def create(self, *, tenant_id: str, customer_id: str | None) -> StoredSession:
        session = StoredSession(
            session_id=f"s{len(self.sessions) + 1}", tenant_id=tenant_id, customer_id=customer_id
        )
        self.sessions[session.session_id] = session
        return session

    async def load(self, session_id: str) -> StoredSession | None:
        return self.sessions.get(session_id)

    async def acquire(self, session_id: str) -> bool:
        if session_id in self.leased:
            return False
        self.leased.add(session_id)
        return True

    async def release(self, session_id: str) -> None:
        self.leased.discard(session_id)

    async def ask_feedback(self, asked: AskedFeedback) -> None:
        self.asked.setdefault(asked.session_id, asked)

    async def rate_session(
        self, session_id: str, *, helpful: bool, comment: str | None
    ) -> AskedFeedback | None:
        asked = self.asked.get(session_id)
        if asked is not None:
            self.ratings[session_id] = (helpful, comment)
        return asked


class SeedOrderRepository:
    """Orders from data/seed with the same ownership rule as the Postgres adapter."""

    def __init__(self) -> None:
        import json
        from pathlib import Path

        seed = Path(__file__).resolve().parents[1] / "data" / "seed" / "orders.json"
        self.orders = {o["order_id"]: Order.model_validate(o) for o in json.loads(seed.read_text())}
        self.lookups: list[tuple[str, str]] = []

    async def get_for_customer(
        self, order_id: str, *, customer_id: str, tenant_id: str
    ) -> Order | None:
        self.lookups.append((order_id, customer_id))
        order = self.orders.get(order_id)
        return order if order and order.customer_id == customer_id else None


def make_orchestrator(
    llm: "ScriptedRouterLLM",
    retriever: FakeRetriever | None = None,
    *,
    orders: SeedOrderRepository | None = None,
    tickets: InMemoryTicketService | None = None,
    audit: InMemoryAuditLog | None = None,
    today: date = date(2026, 10, 5),
    classifier: "PromptAttackClassifier | None" = None,
    max_tickets_per_day: int = 3,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    feedback: "FeedbackPolicy | None" = None,
) -> "ConversationOrchestrator":
    from customer_agent.application.answering import AnswerService
    from customer_agent.application.orchestrator import ConversationOrchestrator
    from customer_agent.application.understanding import UnderstandAgent
    from customer_agent.infrastructure.prompts import load_prompt

    answers = AnswerService(
        retriever or FakeRetriever(documents=[document("policy-returns-v5", 0.7)]),
        llm,
        load_prompt("answer", 4),
        min_similarity=0.35,
    )
    return ConversationOrchestrator(
        UnderstandAgent(llm, load_prompt("router", 5), classifier=classifier),
        answers,
        orders=orders or SeedOrderRepository(),
        tickets=tickets or InMemoryTicketService(),
        audit=audit or InMemoryAuditLog(),
        clock=FixedClock(today),
        tenant_id="nordelektro",
        checkpointer=checkpointer or memory_checkpointer(),
        max_tickets_per_day=max_tickets_per_day,
        feedback=feedback,
    )


def memory_checkpointer() -> InMemorySaver:
    """The production serializer's type allowlist, kept in memory."""
    from customer_agent.application.turn_graph import CHECKPOINT_TYPES

    return InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES))
