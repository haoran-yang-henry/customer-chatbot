"""Intent router: deterministic rules first, then one LLM call for intent + query rewriting."""

import json
import re
import time

from pydantic import BaseModel, ValidationError, field_validator

from customer_agent.application.answering import format_history, search_query
from customer_agent.application.guardrails import check_generated_text
from customer_agent.application.ports import ChatMessage, LLMProvider
from customer_agent.application.prompts import PromptTemplate
from customer_agent.domain.conversation import Conversation
from customer_agent.domain.intents import Intent
from customer_agent.domain.routing import RoutingResult, RoutingSource, extract_order_id
from customer_agent.domain.session import SessionState, WorkflowState
from customer_agent.observability import elapsed_ms, get_logger, step

log = get_logger("router")

_GREETING = re.compile(
    r"^\s*(hallo|hi|hey|servus|moin|guten (morgen|tag|abend)|grüß gott|grüezi|grüss gott)"
    r"[\s!.,]*$",
    re.IGNORECASE,
)
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_MAX_WORDS_FOR_BARE_ORDER_ID = 6
# Short answers to "Soll ich … an unser Service-Team weitergeben?"; anything longer or hedged
# ("ja, aber …") goes to the LLM router, which sees the question in the history.
_MAX_WORDS_FOR_BARE_CONFIRMATION = 6
_YES = re.compile(
    r"^(ja|jawohl|jep|gern|gerne|ok|okay|einverstanden|bitte|klar|sicher|genau|unbedingt"
    r"|machen sie das|mach das|yes)\b",
    re.IGNORECASE,
)
_NO = re.compile(r"^(nein|nö|nee|lieber nicht|nicht nötig|danke,? nein|no)\b", re.IGNORECASE)
_HEDGE = re.compile(r"\b(aber|nicht|nein|doch|später)\b", re.IGNORECASE)
# Reasoning models count hidden reasoning tokens against this budget; 300 produced empty output.
ROUTER_MAX_TOKENS = 800


class _LLMRouting(BaseModel):
    """Shape of the router prompt's JSON output; lenient about types, strict about values."""

    intent: Intent | None = None
    confidence: float = 0.0
    about_specific_order: bool = False
    third_party: bool = False
    money_demand: bool = False
    change_request: bool = False
    confirmation: bool | None = None
    defect_reported: bool = False
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
    if state.workflow is WorkflowState.AWAITING_TICKET_CONFIRMATION:
        confirmation = _bare_confirmation(message)
        if confirmation is not None:
            return RoutingResult(
                intent=state.active_intent,
                confidence=1.0,
                confirmation=confirmation,
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


def _bare_confirmation(message: str) -> bool | None:
    text = message.strip().strip("!.").strip()
    if len(text.split()) > _MAX_WORDS_FOR_BARE_CONFIRMATION:
        return None
    if _NO.match(text):
        return False
    if _YES.match(text) and not _HEDGE.search(text):
        return True
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
        money_demand=parsed.money_demand,
        change_request=parsed.change_request,
        confirmation=parsed.confirmation,
        defect_reported=parsed.defect_reported,
        standalone_query=parsed.standalone_query.strip() or search_query(conversation),
        clarification_question=(parsed.clarification_question or "").strip() or None,
        source=RoutingSource.LLM,
    )


def describe_state(state: SessionState) -> str:
    if state.workflow is WorkflowState.AWAITING_ORDER_ID and state.active_intent:
        return f"The assistant asked for an order number for {state.active_intent}."
    if state.workflow is WorkflowState.AWAITING_TICKET_CONFIRMATION:
        return "The assistant asked whether to forward the customer's request to the service team."
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
        with step("route") as span:
            result = await self._route(conversation, state, session_id=session_id)
            span.set_attributes({"intent": str(result.intent), "source": result.source})
            return result

    async def _route(
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
            result = self._vet_clarification(
                parse_llm_routing(reply.text, conversation), conversation
            )
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

    def _vet_clarification(
        self, result: RoutingResult, conversation: Conversation
    ) -> RoutingResult:
        """The clarification question is LLM output shown verbatim: an injected message could
        turn it into a channel for the system prompt or a phishing link. Unsafe → default."""
        question = result.clarification_question
        if question is None:
            return result
        decision = check_generated_text(
            question, prompt=self._prompt.system, customer=search_query(conversation)
        )
        if decision.blocked:
            log.warning("clarification replaced reasons=%s", ",".join(decision.reasons))
            return result.model_copy(update={"clarification_question": None})
        return result
