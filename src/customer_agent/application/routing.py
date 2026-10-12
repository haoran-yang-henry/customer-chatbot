"""Intent router: deterministic rules first, then one LLM call for intent + query rewriting."""

import json
import re
import time

from pydantic import BaseModel, ValidationError, field_validator

from customer_agent.application.answering import format_history, search_query
from customer_agent.application.ports import ChatMessage, LLMProvider
from customer_agent.application.prompts import PromptTemplate
from customer_agent.domain.conversation import Conversation
from customer_agent.domain.intents import Intent
from customer_agent.domain.routing import RoutingResult, RoutingSource, extract_order_id
from customer_agent.domain.session import SessionState, WorkflowState
from customer_agent.observability import elapsed_ms, get_logger

log = get_logger("router")

_GREETING = re.compile(
    r"^\s*(hallo|hi|hey|servus|moin|guten (morgen|tag|abend)|grüß gott|grüezi|grüss gott)"
    r"[\s!.,]*$",
    re.IGNORECASE,
)
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_MAX_WORDS_FOR_BARE_ORDER_ID = 6
# Reasoning models count hidden reasoning tokens against this budget; 300 produced empty output.
ROUTER_MAX_TOKENS = 800


class _LLMRouting(BaseModel):
    """Shape of the router prompt's JSON output; lenient about types, strict about values."""

    intent: Intent | None = None
    confidence: float = 0.0
    about_specific_order: bool = False
    third_party: bool = False
    standalone_query: str = ""
    clarification_question: str | None = None

    @field_validator("intent", mode="before")
    @classmethod
    def _null_strings(cls, value: object) -> object:
        return None if value in ("", "null", "None") else value

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp(cls, value: object) -> float:
        try:
            return min(max(float(value), 0.0), 1.0)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0.0


def route_by_rules(conversation: Conversation, state: SessionState) -> RoutingResult | None:
    message = conversation.messages[-1].content
    if _GREETING.match(message):
        return RoutingResult(
            intent=Intent.GREETING,
            confidence=1.0,
            standalone_query=message,
            source=RoutingSource.RULE,
        )
    order_id = extract_order_id(message)
    if (
        state.workflow is WorkflowState.AWAITING_ORDER_ID
        and state.active_intent is not None
        and order_id is not None
        and len(message.split()) <= _MAX_WORDS_FOR_BARE_ORDER_ID
    ):
        return RoutingResult(
            intent=state.active_intent,
            confidence=1.0,
            order_id=order_id,
            about_specific_order=True,
            standalone_query=search_query(conversation),
            source=RoutingSource.RULE,
        )
    return None


def parse_llm_routing(text: str, conversation: Conversation) -> RoutingResult:
    """Never raises: unusable output becomes a FALLBACK result (answered as knowledge)."""
    message = conversation.messages[-1].content
    match = _JSON_OBJECT.search(text)
    try:
        parsed = _LLMRouting.model_validate(json.loads(match.group(0)) if match else {})
    except (json.JSONDecodeError, ValidationError):
        parsed = None
    if parsed is None or not match:
        return RoutingResult(
            intent=None,
            confidence=0.0,
            order_id=extract_order_id(message),
            standalone_query=search_query(conversation),
            source=RoutingSource.FALLBACK,
        )
    order_id = extract_order_id(message)
    return RoutingResult(
        intent=parsed.intent,
        confidence=parsed.confidence,
        order_id=order_id,
        about_specific_order=parsed.about_specific_order or order_id is not None,
        third_party=parsed.third_party,
        standalone_query=parsed.standalone_query.strip() or search_query(conversation),
        clarification_question=(parsed.clarification_question or "").strip() or None,
        source=RoutingSource.LLM,
    )


def describe_state(state: SessionState) -> str:
    if state.workflow is WorkflowState.AWAITING_ORDER_ID and state.active_intent:
        return f"The assistant asked for an order number for {state.active_intent}."
    if state.active_intent:
        return f"Previous intent: {state.active_intent}."
    return "New conversation."


class IntentRouter:
    def __init__(self, llm: LLMProvider, prompt: PromptTemplate) -> None:
        self._llm = llm
        self._prompt = prompt

    async def route(
        self, conversation: Conversation, state: SessionState, *, session_id: str
    ) -> RoutingResult:
        start = time.perf_counter()
        result = route_by_rules(conversation, state)
        if result is None:
            reply = await self._llm.complete(
                [
                    ChatMessage(role="system", content=self._prompt.system),
                    ChatMessage(
                        role="user",
                        content=self._prompt.render_user(
                            state=describe_state(state),
                            history=format_history(conversation),
                            message=conversation.messages[-1].content,
                        ),
                    ),
                ],
                max_tokens=ROUTER_MAX_TOKENS,
                temperature=0.0,
                session_id=session_id,
            )
            result = parse_llm_routing(reply.text, conversation)
        log.info(
            "source=%s intent=%s confidence=%.2f order_id=%s specific=%s took=%dms",
            result.source,
            result.intent,
            result.confidence,
            result.order_id or "-",
            result.about_specific_order,
            elapsed_ms(start),
        )
        log.debug("standalone_query=%r", result.standalone_query)
        return result
