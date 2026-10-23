"""Grounded answering (Day 3): retrieve → relevance gate → answer agent.

The LLM is only called when retrieval found sufficiently similar evidence; otherwise the answer is
an honest "no answer" (FR-12). The answer comes from a tool-less LangChain agent without a
checkpointer (rebuild plan §4.5); its middleware writes the prompt and checks the output:

    before_model   UntrustedEvidence          community documents quoted as data (T02)
    model call     AnswerPrompt               evidence, earlier turns and question
    after_model    OutputGuard → OutputPII    citations out, output check, payment data masked

The hooks only record their verdict; the response is built from the agent's final state, after
the complete model output, so the streaming and non-streaming paths make identical decisions.
"""

import hashlib
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, NotRequired, cast

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from pydantic import BaseModel, ConfigDict

from customer_agent.application.citations import (
    StreamingCitationFilter,
    extract_citations,
    is_no_answer,
)
from customer_agent.application.fallbacks import fallback
from customer_agent.application.guardrails import Evidence
from customer_agent.application.guardrails.evidence import quote_untrusted
from customer_agent.application.guardrails.output import (
    OutputScreen,
    mask_output,
    output_decision,
    screen_output,
)
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
class AnswerContext:
    """What the hooks need besides the question: the retrieved documents, what the answer may
    repeat, the earlier turns (masked, formatted) and the session id. Runtime context, so it is
    neither state nor traced as input."""

    documents: tuple[RetrievedDocument, ...]
    evidence: Evidence
    history: str
    session_id: str


class AnswerState(AgentState[Any]):
    evidence_text: NotRequired[str]
    declined: NotRequired[bool]
    cited: NotRequired[list[str]]
    answer_text: NotRequired[str]
    output_reasons: NotRequired[list[str]]
    output_blocked: NotRequired[bool]


Middleware = AgentMiddleware[AnswerState, AnswerContext, Any]


def _question(state: AnswerState) -> str:
    """The customer's (masked) message: the agent's only human message."""
    return next(m for m in state["messages"] if isinstance(m, HumanMessage)).text


def _screen(state: AnswerState) -> OutputScreen:
    return OutputScreen(
        text=state.get("answer_text", ""),
        reasons=tuple(state.get("output_reasons", ())),
        blocked=state.get("output_blocked", False),
    )


def _record(screen: OutputScreen) -> dict[str, Any]:
    return {
        "answer_text": screen.text,
        "output_reasons": list(screen.reasons),
        "output_blocked": screen.blocked,
    }


class UntrustedEvidence(Middleware):
    """Retrieval guardrail: untrusted documents are quoted, instruction-like lines dropped."""

    state_schema = AnswerState

    def before_model(
        self, state: AnswerState, runtime: Runtime[AnswerContext]
    ) -> dict[str, Any] | None:
        return {"evidence_text": format_evidence(list(runtime.context.documents))}


class AnswerPrompt(Middleware):
    """The answer request; byte for byte the request of the hand-built service, so recordings
    stay valid."""

    state_schema = AnswerState

    def __init__(self, prompt: PromptTemplate, max_tokens: int) -> None:
        super().__init__()
        self._prompt = prompt
        self._max_tokens = max_tokens

    async def awrap_model_call(
        self,
        request: ModelRequest[AnswerContext],
        handler: Callable[[ModelRequest[AnswerContext]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]:
        state = cast(AnswerState, request.state)
        context = request.runtime.context
        system, user = self._prompt.render_messages(
            evidence=state["evidence_text"], history=context.history, question=_question(state)
        )
        settings = {
            **request.model_settings,
            "max_tokens": self._max_tokens,
            "temperature": TEMPERATURE,
            "session_id": context.session_id,
        }
        return await handler(
            request.override(
                system_message=cast(SystemMessage, system),
                messages=[cast(HumanMessage, user)],
                model_settings=settings,
            )
        )


class OutputGuard(Middleware):
    """The no-answer token declines; otherwise the citations are taken out and the answer is
    checked: cited, German, no leak, nothing unverifiable (ADR-0011)."""

    state_schema = AnswerState

    def __init__(self, prompt: PromptTemplate) -> None:
        super().__init__()
        self._prompt = prompt

    def after_model(
        self, state: AnswerState, runtime: Runtime[AnswerContext]
    ) -> dict[str, Any] | None:
        raw = state["messages"][-1].text
        if is_no_answer(raw):
            return {"declined": True}
        answer, cited = extract_citations(raw)
        offered = {d.document_id for d in runtime.context.documents}
        with step("output_guardrail"):
            screen = screen_output(
                answer,
                cited,
                offered,
                prompt=self._prompt.system,
                evidence=runtime.context.evidence,
            )
        return {"cited": cited, **_record(screen)}


class OutputPII(Middleware):
    """The output side of ``GermanPII``: IBAN and card numbers in the answer masked."""

    state_schema = AnswerState

    def after_model(
        self, state: AnswerState, runtime: Runtime[AnswerContext]
    ) -> dict[str, Any] | None:
        return None if state.get("declined") else _record(mask_output(_screen(state)))


def _response(state: AnswerState, context: AnswerContext) -> AgentResponse:
    if state.get("declined"):
        log.info("no_answer reason=model_declined")
        return fallback(["model_declined"])
    decision = output_decision(_screen(state))
    record_guardrail(decision)
    if decision.blocked:
        log.warning("output blocked reasons=%s", ",".join(decision.reasons))
        return fallback(decision.reasons)
    if decision.action is GuardrailAction.REDACT:
        log.warning("output redacted reasons=%s", ",".join(decision.reasons))
    offered = {d.document_id for d in context.documents}
    sources = [c for c in state.get("cited", []) if c in offered]
    answer = state.get("answer_text", "")
    log.info("answer sources=%s chars=%d", ",".join(sources), len(answer))
    return AgentResponse(
        answer=answer, behavior=Behavior.ANSWER, sources=sources, reasons=decision.reasons
    )


class AnswerAgent:
    def __init__(self, model: BaseChatModel, prompt: PromptTemplate, *, max_tokens: int) -> None:
        middleware: list[Middleware] = [
            UntrustedEvidence(),
            AnswerPrompt(prompt, max_tokens),
            # after_model hooks run last to first: OutputGuard, then OutputPII.
            OutputPII(),
            OutputGuard(prompt),
        ]
        self._agent = create_agent(
            model,
            tools=None,
            middleware=middleware,
            context_schema=AnswerContext,
            name="answer",
        )

    async def run(self, question: str, context: AnswerContext) -> AgentResponse:
        state = await self._agent.ainvoke({"messages": [HumanMessage(question)]}, context=context)
        return _response(cast(AnswerState, state), context)

    async def stream(self, question: str, context: AnswerContext) -> AsyncIterator[StreamEvent]:
        """The model's tokens without citation markers, then the checked answer, which
        replaces them."""
        citations = StreamingCitationFilter()
        question_message = HumanMessage(question)
        state = AnswerState(messages=[question_message])  # "values" reports the input first
        parts = self._agent.astream(
            {"messages": [question_message]}, context=context, stream_mode=["messages", "values"]
        )
        async for mode, data in cast(AsyncIterator[tuple[str, Any]], parts):
            if mode == "values":
                state = cast(AnswerState, data)
                continue
            chunk, metadata = cast(tuple[AIMessageChunk, dict[str, Any]], data)
            if metadata.get("langgraph_node") == "model" and (text := citations.feed(chunk.text)):
                yield TextDelta(text=text)
        if text := citations.flush():
            yield TextDelta(text=text)
        yield FinalAnswer(response=_response(state, context))


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
        self._agent = AnswerAgent(llm, prompt, max_tokens=max_tokens)
        self._clock = clock
        self._tenant_id = tenant_id
        self._min_similarity = min_similarity
        self._top_k = top_k

    async def _prepare(
        self, conversation: Conversation, query: str | None, session: str | None
    ) -> AnswerContext | AgentResponse:
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
        return AnswerContext(
            documents=tuple(documents),
            evidence=answer_evidence(documents, conversation),
            history=format_history(conversation),
            session_id=session or stateless_session_id(conversation),
        )

    async def answer(
        self,
        conversation: Conversation,
        *,
        query: str | None = None,
        session_id: str | None = None,
    ) -> AgentResponse:
        """``query``: the router's standalone question; ``None`` → joined user turns."""
        context = await self._prepare(conversation, query, session_id)
        if isinstance(context, AgentResponse):
            return context
        return await self._agent.run(conversation.messages[-1].content, context)

    async def answer_stream(
        self,
        conversation: Conversation,
        *,
        query: str | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        context = await self._prepare(conversation, query, session_id)
        if isinstance(context, AgentResponse):
            yield FinalAnswer(response=context)
            return
        async for event in self._agent.stream(conversation.messages[-1].content, context):
            yield event
