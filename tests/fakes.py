"""Test doubles for D3 ports."""

from collections.abc import AsyncIterator, Sequence
from datetime import date

from customer_agent.application.ports import (
    ChatMessage,
    LLMResult,
    ProviderUnavailableError,
    StoredSession,
)
from customer_agent.domain.conversation import AgentResponse, Message, Role
from customer_agent.domain.knowledge import SourceType, TrustLevel
from customer_agent.domain.retrieval import (
    ChunkHit,
    RetrievedDocument,
    SearchMode,
    SearchScope,
)
from customer_agent.domain.session import SessionState


def document(doc_id: str, similarity: float | None, content: str = "Inhalt.") -> RetrievedDocument:
    hit = ChunkHit(
        chunk_id=hash(doc_id) % 10_000,
        document_id=doc_id,
        title=doc_id,
        source_type=SourceType.POLICY,
        trust_level=TrustLevel.OFFICIAL,
        content=content,
        score=0.03,
        similarity=similarity,
    )
    return RetrievedDocument(
        document_id=doc_id,
        title=doc_id,
        source_type=SourceType.POLICY,
        trust_level=TrustLevel.OFFICIAL,
        score=0.03,
        chunks=[hit],
    )


class FakeRetriever:
    def __init__(self, documents: list[RetrievedDocument]) -> None:
        self.documents = documents
        self.queries: list[tuple[str, SearchScope]] = []

    async def retrieve(
        self,
        query: str,
        scope: SearchScope,
        *,
        top_k: int = 5,
        mode: SearchMode = SearchMode.HYBRID,
    ) -> list[RetrievedDocument]:
        self.queries.append((query, scope))
        return self.documents[:top_k]


class FakeLLM:
    """Returns a scripted reply; streams it in small pieces to exercise the stream filter."""

    def __init__(self, reply: str = "", *, fail: bool = False, piece: int = 7) -> None:
        self.reply = reply
        self.fail = fail
        self.piece = piece
        self.calls: list[list[ChatMessage]] = []
        self.session_ids: list[str | None] = []

    @property
    def model_name(self) -> str:
        return "fake-llm"

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
        session_id: str | None = None,
    ) -> LLMResult:
        self.calls.append(list(messages))
        self.session_ids.append(session_id)
        if self.fail:
            raise ProviderUnavailableError("fake outage")
        return LLMResult(text=self.reply, model="fake-llm")

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
        session_id: str | None = None,
    ) -> AsyncIterator[str]:
        result = await self.complete(
            messages, max_tokens=max_tokens, temperature=temperature, session_id=session_id
        )
        for i in range(0, len(result.text), self.piece):
            yield result.text[i : i + self.piece]


class FixedClock:
    def __init__(self, today: date) -> None:
        self._today = today

    def today(self) -> date:
        return self._today


class ScriptedRouterLLM(FakeLLM):
    """Answers router prompts with a JSON routing and answer prompts with ``answer``."""

    def __init__(self, routing: str, answer: str = "Ok. [policy-returns-v5]") -> None:
        super().__init__(answer)
        self.routing = routing
        self.router_calls = 0
        self.router_prompts: list[list[ChatMessage]] = []
        self.router_session_ids: list[str | None] = []

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
        session_id: str | None = None,
    ) -> LLMResult:
        if "You classify customer messages" in messages[0].content:
            self.router_calls += 1
            self.router_prompts.append(list(messages))
            self.router_session_ids.append(session_id)
            if self.fail:
                raise ProviderUnavailableError("fake outage")
            return LLMResult(text=self.routing, model="fake-llm")
        return await super().complete(
            messages, max_tokens=max_tokens, temperature=temperature, session_id=session_id
        )


class InMemorySessionStore:
    def __init__(self) -> None:
        self.sessions: dict[str, StoredSession] = {}

    async def create(self, *, tenant_id: str, customer_id: str | None) -> StoredSession:
        session = StoredSession(
            session_id=f"s{len(self.sessions) + 1}",
            customer_id=customer_id,
            state=SessionState(),
            messages=[],
        )
        self.sessions[session.session_id] = session
        return session

    async def load(self, session_id: str) -> StoredSession | None:
        return self.sessions.get(session_id)

    async def save_turn(
        self, session_id: str, *, state: SessionState, user: str, reply: AgentResponse
    ) -> None:
        old = self.sessions[session_id]
        self.sessions[session_id] = old.model_copy(
            update={
                "state": state,
                "messages": [
                    *old.messages,
                    Message(role=Role.USER, content=user),
                    Message(role=Role.ASSISTANT, content=reply.answer),
                ],
            }
        )
