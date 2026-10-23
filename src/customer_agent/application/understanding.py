"""Stage 1 of a turn (rebuild plan §4.2, ADR-0015): understand the message.

One tool-less LangChain agent without a checkpointer runs the guardrails as middleware
(ADR-0011 layers, rebuild plan §4.5) and the router as its only model call:

    before_agent   InputSanitizer → InjectionSignals → GermanPII     the input check, step by step
    before_model   InputGate, RuleRouting                             end without a model call
    model call     ProviderOutage ⟶ AttackClassifier ⟶ RouterPrompt   classifier concurrent
    after_model    RouterOutput                                       parse and vet the routing

The input hooks only record their verdict; ``InputGate`` ends a blocked turn. So a blocked
message is masked like any other, and the injection signals are scored on the unmasked text,
exactly as ``check_input`` does. The raw message exists only in this call: there is no
checkpointer, and Langfuse's mask hook covers the traced input.
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, NotRequired, cast

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
    hook_config,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

from customer_agent.application.guardrails.classifier import PromptAttackClassifier
from customer_agent.application.guardrails.input import (
    InputScreen,
    input_decision,
    mask_input,
    sanitize,
    screen_injection,
)
from customer_agent.application.ports import ProviderUnavailableError
from customer_agent.application.prompts import PromptTemplate
from customer_agent.application.routing import (
    ROUTER_MAX_TOKENS,
    log_routing,
    parse_llm_routing,
    route_by_rules,
    router_messages,
    vet_clarification,
)
from customer_agent.domain.conversation import Conversation, Message, Role
from customer_agent.domain.guardrails import GuardrailDecision
from customer_agent.domain.routing import RoutingResult
from customer_agent.domain.session import SessionState
from customer_agent.observability import get_logger, step

log = get_logger("understand")


@dataclass(frozen=True)
class UnderstandContext:
    """What the hooks need besides the message: earlier turns (already masked), the session's
    state and ids. Runtime context, so it is neither state nor traced as input."""

    history: tuple[Message, ...]
    state: SessionState
    session_id: str
    customer_id: str | None


class UnderstandState(AgentState[Any]):
    # Only verdicts and masked text: state is traced, and a dataclass would bypass the mask hook.
    hidden_text: NotRequired[str]
    input_reasons: NotRequired[list[str]]
    input_blocked: NotRequired[bool]
    input_scores: NotRequired[dict[str, float]]
    routing: NotRequired[RoutingResult]
    verdict: NotRequired[GuardrailDecision]
    outage: NotRequired[bool]


@dataclass(frozen=True)
class Understanding:
    """The input check, and unless it blocked: the routing and the classifier's verdict."""

    input: GuardrailDecision
    routing: RoutingResult | None = None
    verdict: GuardrailDecision | None = None
    outage: bool = False  # the router's provider failed: nothing was understood

    @property
    def text(self) -> str:
        """The message as the input check left it (masked); the only version used from here."""
        return self.input.content or ""


Middleware = AgentMiddleware[UnderstandState, UnderstandContext, Any]


def _message(state: UnderstandState) -> HumanMessage:
    """The customer's message: the agent's only human message."""
    return next(m for m in state["messages"] if isinstance(m, HumanMessage))


def _screen(state: UnderstandState) -> InputScreen:
    return InputScreen(
        text=_message(state).text,
        hidden=state.get("hidden_text", ""),
        reasons=tuple(state.get("input_reasons", ())),
        blocked=state.get("input_blocked", False),
        scores=state.get("input_scores", {}),
    )


def _record(state: UnderstandState, screen: InputScreen) -> dict[str, Any]:
    message = _message(state).model_copy(update={"content": screen.text})  # same id: replaced
    return {
        "messages": [message],
        "hidden_text": screen.hidden,
        "input_reasons": list(screen.reasons),
        "input_blocked": screen.blocked,
        "input_scores": dict(screen.scores),
    }


def _conversation(state: UnderstandState, context: UnderstandContext) -> Conversation:
    latest = Message(role=Role.USER, content=_message(state).text)
    return Conversation(messages=[*context.history, latest], customer_id=context.customer_id)


class InputSanitizer(Middleware):
    """NFKC, invisible characters out; an empty or over-long message is blocked."""

    state_schema = UnderstandState

    def before_agent(
        self, state: UnderstandState, runtime: Runtime[UnderstandContext]
    ) -> dict[str, Any] | None:
        return _record(state, sanitize(_message(state).text))


class InjectionSignals(Middleware):
    """Weighted German and English injection signals on the unmasked text."""

    state_schema = UnderstandState

    def before_agent(
        self, state: UnderstandState, runtime: Runtime[UnderstandContext]
    ) -> dict[str, Any] | None:
        return _record(state, screen_injection(_screen(state)))


class GermanPII(Middleware):
    """Personal data masked with German placeholders before anything reads the message (NFR-12).
    The hook of LangChain's ``PIIMiddleware``; that class has English placeholders, no IBAN,
    phone or secret detectors and does not report what it found (rebuild plan §4.5)."""

    state_schema = UnderstandState

    def before_agent(
        self, state: UnderstandState, runtime: Runtime[UnderstandContext]
    ) -> dict[str, Any] | None:
        return _record(state, mask_input(_screen(state)))


class InputGate(Middleware):
    """A blocked message ends the turn before any model call."""

    state_schema = UnderstandState

    @hook_config(can_jump_to=["end"])
    def before_model(
        self, state: UnderstandState, runtime: Runtime[UnderstandContext]
    ) -> dict[str, Any] | None:
        return {"jump_to": "end"} if state.get("input_blocked") else None


class RuleRouting(Middleware):
    """A greeting, an answer to the ticket question or a bare order number needs no model; the
    classifier is skipped as well."""

    state_schema = UnderstandState

    @hook_config(can_jump_to=["end"])
    def before_model(
        self, state: UnderstandState, runtime: Runtime[UnderstandContext]
    ) -> dict[str, Any] | None:
        routing = route_by_rules(_conversation(state, runtime.context), runtime.context.state)
        if routing is None:
            return None
        log_routing(routing)
        return {"routing": routing, "jump_to": "end"}


class ProviderOutage(Middleware):
    """The router's provider failed: the turn continues with the masked message and no routing,
    so the orchestrator can answer with the contact channels (NFR-11)."""

    state_schema = UnderstandState

    async def awrap_model_call(
        self,
        request: ModelRequest[UnderstandContext],
        handler: Callable[[ModelRequest[UnderstandContext]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]:
        try:
            return await handler(request)
        except ProviderUnavailableError as exc:
            log.error("provider unavailable: %s", exc)
            return ExtendedModelResponse(
                model_response=ModelResponse(result=[AIMessage(content="")]),
                command=Command(update={"outage": True}),
            )


class AttackClassifier(Middleware):
    """The LLM attack classifier (ADR-0011) concurrent with the router call: it costs a call,
    not latency. Fails open inside ``PromptAttackClassifier``."""

    state_schema = UnderstandState

    def __init__(self, classifier: PromptAttackClassifier) -> None:
        super().__init__()
        self._classifier = classifier

    async def awrap_model_call(
        self,
        request: ModelRequest[UnderstandContext],
        handler: Callable[[ModelRequest[UnderstandContext]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]:
        context = request.runtime.context
        conversation = _conversation(cast(UnderstandState, request.state), context)
        verdict = asyncio.create_task(
            self._classifier.classify(conversation, session_id=context.session_id)
        )
        try:
            response = await handler(request)
        except BaseException:
            verdict.cancel()
            raise
        return ExtendedModelResponse(
            model_response=response, command=Command(update={"verdict": await verdict})
        )


class RouterPrompt(Middleware):
    """The router's request, rendered from the masked message, the earlier turns and the state;
    byte for byte the request of the hand-built router, so recordings stay valid. The call is
    the ``route`` step: the load report subtracts it as model time (NFR-09)."""

    state_schema = UnderstandState

    def __init__(self, prompt: PromptTemplate) -> None:
        super().__init__()
        self._prompt = prompt

    async def awrap_model_call(
        self,
        request: ModelRequest[UnderstandContext],
        handler: Callable[[ModelRequest[UnderstandContext]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]:
        context = request.runtime.context
        conversation = _conversation(cast(UnderstandState, request.state), context)
        system, user = router_messages(self._prompt, conversation, context.state)
        settings = {
            **request.model_settings,
            "max_tokens": ROUTER_MAX_TOKENS,
            "temperature": 0.0,
            "session_id": context.session_id,
        }
        request = request.override(
            system_message=cast(SystemMessage, system),
            messages=[cast(HumanMessage, user)],
            model_settings=settings,
        )
        with step("route"):
            return await handler(request)


class RouterOutput(Middleware):
    """Parse the router's JSON leniently (FALLBACK when unusable; order numbers from the regex
    only) and vet its clarification question before anyone shows it."""

    state_schema = UnderstandState

    def __init__(self, prompt: PromptTemplate) -> None:
        super().__init__()
        self._prompt = prompt

    def after_model(
        self, state: UnderstandState, runtime: Runtime[UnderstandContext]
    ) -> dict[str, Any] | None:
        if state.get("outage"):
            return None
        conversation = _conversation(state, runtime.context)
        reply = state["messages"][-1]
        routing = parse_llm_routing(reply.text, conversation)
        routing = vet_clarification(routing, conversation, self._prompt)
        log_routing(routing)
        return {"routing": routing}


class UnderstandAgent:
    def __init__(
        self,
        model: BaseChatModel,
        prompt: PromptTemplate,
        *,
        classifier: PromptAttackClassifier | None = None,
    ) -> None:
        middleware: list[Middleware] = [
            InputSanitizer(),
            InjectionSignals(),
            GermanPII(),
            InputGate(),
            RuleRouting(),
            ProviderOutage(),
            *([AttackClassifier(classifier)] if classifier is not None else []),
            RouterPrompt(prompt),
            RouterOutput(prompt),
        ]
        self._agent = create_agent(
            model,
            tools=None,
            middleware=middleware,
            context_schema=UnderstandContext,
            name="understand",
        )

    async def run(
        self, conversation: Conversation, state: SessionState, *, session_id: str
    ) -> Understanding:
        context = UnderstandContext(
            history=tuple(conversation.messages[:-1]),
            state=state,
            session_id=session_id,
            customer_id=conversation.customer_id,
        )
        result = await self._agent.ainvoke(
            {"messages": [HumanMessage(conversation.messages[-1].content)]}, context=context
        )
        outcome = cast(UnderstandState, result)
        screen = InputScreen(
            text=_message(outcome).text,
            reasons=tuple(outcome.get("input_reasons", ())),
            blocked=outcome.get("input_blocked", False),
            scores=outcome.get("input_scores", {}),
        )
        return Understanding(
            input=input_decision(screen),
            routing=outcome.get("routing"),
            verdict=outcome.get("verdict"),
            outage=outcome.get("outage", False),
        )
