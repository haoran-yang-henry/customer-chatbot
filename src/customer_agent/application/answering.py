"""Grounded answering (Day 3): retrieve → relevance gate → prompt → LLM → citations → output check.

The LLM is only called when retrieval found sufficiently similar evidence; otherwise the answer is
an honest "no answer" (FR-12). The final answer always comes from the complete model output, so
the streaming and non-streaming paths make identical decisions.
"""

import hashlib
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ConfigDict

from customer_agent.application.citations import (
    StreamingCitationFilter,
    extract_citations,
    is_no_answer,
)
from customer_agent.application.fallbacks import fallback
from customer_agent.application.guardrails import Evidence, check_output
from customer_agent.application.guardrails.evidence import quote_untrusted
from customer_agent.application.ports import Clock, Retriever
from customer_agent.application.prompts import PromptTemplate
from customer_agent.domain.conversation import AgentResponse, Behavior, Conversation, Role
from customer_agent.domain.guardrails import GuardrailAction
from customer_agent.domain.knowledge import TrustLevel
from customer_agent.domain.retrieval import RetrievedDocument, SearchScope
from customer_agent.observability import elapsed_ms, get_logger, record_guardrail, step

log = get_logger("answer")

MAX_HISTORY_TURNS = 6
TEMPERATURE = 0.1


class TextDelta(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str


class FinalAnswer(BaseModel):
    """Authoritative result; replaces anything streamed before (e.g. after a guardrail block)."""

    model_config = ConfigDict(frozen=True)

    response: AgentResponse


StreamEvent = TextDelta | FinalAnswer


def search_query(conversation: Conversation) -> str:
    """All user turns joined. Fallback when the router provides no standalone query."""
    return " ".join(m.content for m in conversation.messages if m.role is Role.USER)


def stateless_session_id(conversation: Conversation) -> str:
    """Stable id for stateless callers (evaluation): hash of the conversation's first message.
    Served conversations use the persisted session id instead."""
    seed = f"{conversation.customer_id or '-'}|{conversation.messages[0].content}"
    return hashlib.sha256(seed.encode()).hexdigest()[:32]


def _text(doc: RetrievedDocument) -> str:
    return "\n\n".join(chunk.content for chunk in doc.chunks)


def format_evidence(documents: list[RetrievedDocument]) -> str:
    """Untrusted documents are quoted, with instruction-like lines dropped (T02, Day 6)."""
    blocks = []
    for doc in documents:
        text = _text(doc)
        if doc.trust_level is TrustLevel.UNTRUSTED:
            text, dropped = quote_untrusted(text)
            if dropped:
                log.warning("untrusted document=%s dropped_lines=%d", doc.document_id, dropped)
        blocks.append(
            f'<document id="{doc.document_id}" trust="{doc.trust_level}">\n{text}\n</document>'
        )
    return "\n\n".join(blocks)


def answer_evidence(documents: list[RetrievedDocument], conversation: Conversation) -> Evidence:
    """What the answer may repeat verbatim (links, codes, numbers): see guardrails.output."""
    trusted = [d for d in documents if d.trust_level is not TrustLevel.UNTRUSTED]
    untrusted = [d for d in documents if d.trust_level is TrustLevel.UNTRUSTED]
    return Evidence(
        trusted="\n".join(_text(d) for d in trusted),
        untrusted="\n".join(_text(d) for d in untrusted),
        customer=search_query(conversation),
    )


def format_history(conversation: Conversation) -> str:
    earlier = conversation.messages[:-1][-MAX_HISTORY_TURNS:]
    if not earlier:
        return "(none)"
    labels = {Role.USER: "Customer", Role.ASSISTANT: "Assistant"}
    return "\n".join(f"{labels[m.role]}: {m.content}" for m in earlier)


@dataclass(frozen=True)
class _Prepared:
    documents: list[RetrievedDocument]
    evidence: Evidence
    messages: list[BaseMessage]
    session_id: str


class AnswerService:
    def __init__(
        self,
        retriever: Retriever,
        llm: BaseChatModel,
        prompt: PromptTemplate,
        clock: Clock,
        *,
        tenant_id: str,
        min_similarity: float,
        top_k: int = 4,
        max_tokens: int = 800,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._prompt = prompt
        self._clock = clock
        self._tenant_id = tenant_id
        self._min_similarity = min_similarity
        self._top_k = top_k
        self._max_tokens = max_tokens

    async def _prepare(
        self, conversation: Conversation, query: str | None, session: str | None
    ) -> _Prepared | AgentResponse:
        start = time.perf_counter()
        scope = SearchScope(today=self._clock.today(), tenant_id=self._tenant_id)
        documents = await self._retriever.retrieve(
            query or search_query(conversation), scope, top_k=self._top_k
        )
        best = max((d.similarity or 0.0 for d in documents), default=0.0)
        log.info(
            "retrieved=%s best_similarity=%.3f took=%dms",
            ",".join(d.document_id for d in documents) or "-",
            best,
            elapsed_ms(start),
        )
        if best < self._min_similarity:
            log.info("no_answer reason=low_similarity threshold=%.2f", self._min_similarity)
            return fallback(["low_similarity"])

        question = conversation.messages[-1].content
        messages = self._prompt.render_messages(
            evidence=format_evidence(documents),
            history=format_history(conversation),
            question=question,
        )
        return _Prepared(
            documents=documents,
            evidence=answer_evidence(documents, conversation),
            messages=messages,
            session_id=session or stateless_session_id(conversation),
        )

    def _finish(self, raw: str, prepared: _Prepared) -> AgentResponse:
        if is_no_answer(raw):
            log.info("no_answer reason=model_declined")
            return fallback(["model_declined"])
        offered = {d.document_id for d in prepared.documents}
        answer, cited = extract_citations(raw)
        with step("output_guardrail"):
            decision = check_output(
                answer, cited, offered, prompt=self._prompt.system, evidence=prepared.evidence
            )
        record_guardrail(decision)
        if decision.blocked:
            log.warning("output blocked reasons=%s", ",".join(decision.reasons))
            return fallback(decision.reasons)
        if decision.action is GuardrailAction.REDACT and decision.content is not None:
            log.warning("output redacted reasons=%s", ",".join(decision.reasons))
            answer = decision.content
        sources = [c for c in cited if c in offered]
        log.info("answer sources=%s chars=%d", ",".join(sources), len(answer))
        return AgentResponse(
            answer=answer, behavior=Behavior.ANSWER, sources=sources, reasons=decision.reasons
        )

    async def answer(
        self,
        conversation: Conversation,
        *,
        query: str | None = None,
        session_id: str | None = None,
    ) -> AgentResponse:
        """``query``: the router's standalone question; ``None`` → joined user turns."""
        prepared = await self._prepare(conversation, query, session_id)
        if isinstance(prepared, AgentResponse):
            return prepared
        reply = await self._llm.ainvoke(
            prepared.messages,
            session_id=prepared.session_id,
            max_tokens=self._max_tokens,
            temperature=TEMPERATURE,
        )
        return self._finish(reply.text, prepared)

    async def answer_stream(
        self,
        conversation: Conversation,
        *,
        query: str | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        prepared = await self._prepare(conversation, query, session_id)
        if isinstance(prepared, AgentResponse):
            yield FinalAnswer(response=prepared)
            return
        stream_filter = StreamingCitationFilter()
        async for chunk in self._llm.astream(
            prepared.messages,
            session_id=prepared.session_id,
            max_tokens=self._max_tokens,
            temperature=TEMPERATURE,
        ):
            if text := stream_filter.feed(chunk.text):
                yield TextDelta(text=text)
        if text := stream_filter.flush():
            yield TextDelta(text=text)
        yield FinalAnswer(response=self._finish(stream_filter.raw, prepared))
