"""Test doubles for D3 ports."""

from collections.abc import AsyncIterator, Sequence
from datetime import date

from customer_agent.application.ports import ChatMessage, LLMResult, ProviderUnavailableError
from customer_agent.domain.knowledge import SourceType, TrustLevel
from customer_agent.domain.retrieval import (
    ChunkHit,
    RetrievedDocument,
    SearchMode,
    SearchScope,
)


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
