"""Conversation orchestrator: the ``CustomerServiceAgent`` behind the API and the evaluation.

One turn in two stages (rebuild plan §4.2): the understand agent checks the message and routes
it (input guardrail, rule routing, intent router ‖ attack classifier); a blocked message ends
the turn there. Otherwise the turn graph decides and acts on the masked message (``decide``,
order guidance, tickets, the grounded answer) and streams the answer's tokens.
"""

import re
from collections.abc import AsyncIterator
from typing import Any, cast

from pydantic import BaseModel, ConfigDict

from customer_agent.application.answering import (
    AnswerService,
    TextDelta,
    stateless_session_id,
)
from customer_agent.application.fallbacks import (
    PAYMENT_DATA_NOTICE,
    TICKET_QUESTION,
    fallback,
    intent_of_question,
    outage_contact_reply,
    with_payment_data_notice,
)
from customer_agent.application.guardrails import PAYMENT_DATA_REASONS
from customer_agent.application.ports import (
    AuditLog,
    Clock,
    OrderRepository,
    ProviderUnavailableError,
    TicketService,
)
from customer_agent.application.tools import CreateTicket, GetOrderStatus
from customer_agent.application.turn_graph import TurnContext, TurnGraph, TurnState
from customer_agent.application.understanding import UnderstandAgent, Understanding
from customer_agent.domain.conversation import AgentResponse, Conversation, Message, Role
from customer_agent.domain.guardrails import GuardrailDecision
from customer_agent.domain.intents import Intent
from customer_agent.domain.routing import extract_order_id
from customer_agent.domain.session import SessionState, WorkflowState
from customer_agent.observability import get_logger, record_guardrail
from customer_agent.observability import step as agent_step
from customer_agent.observability.metrics import AGENT_REPLIES

log = get_logger("chat")

_TICKET_NUMBER = re.compile(r"Ticketnummer (?:lautet )?(T-\d{6})")


class TurnResult(BaseModel):
    """The agent's reply plus what to persist for the next turn: the session state and the
    customer message as the guardrail left it (personal data masked, never the raw text)."""

    model_config = ConfigDict(frozen=True)

    response: AgentResponse
    state: SessionState
    user_text: str


TurnEvent = TextDelta | TurnResult


class ConversationOrchestrator:
    def __init__(
        self,
        understand: UnderstandAgent,
        answers: AnswerService,
        *,
        orders: OrderRepository,
        tickets: TicketService,
        audit: AuditLog,
        clock: Clock,
        tenant_id: str,
        max_tickets_per_day: int | None = 3,  # None: no limit
    ) -> None:
        self._understand = understand
        self._graph = TurnGraph(
            answers,
            GetOrderStatus(orders=orders, audit_log=audit),
            CreateTicket(tickets=tickets, audit_log=audit, max_per_day=max_tickets_per_day),
            clock,
        ).graph
        self._tenant_id = tenant_id

    # --- CustomerServiceAgent (stateless; used by the evaluation) -------------------------

    async def respond(self, conversation: Conversation) -> AgentResponse:
        turn = await self.handle(
            conversation, SessionState(), session_id=stateless_session_id(conversation)
        )
        return turn.response

    # --- stateful turns (used by ChatService) -----------------------------------------------

    async def handle(
        self, conversation: Conversation, state: SessionState, *, session_id: str
    ) -> TurnResult:
        async for event in self.handle_stream(conversation, state, session_id=session_id):
            if isinstance(event, TurnResult):
                return event
        raise RuntimeError("turn ended without a result")  # pragma: no cover

    async def handle_stream(
        self, conversation: Conversation, state: SessionState, *, session_id: str
    ) -> AsyncIterator[TurnEvent]:
        recovered = recover_state(conversation, state)
        with agent_step("understand"):
            understood = await self._understand.run(conversation, recovered, session_id=session_id)
        screen = understood.input
        record_guardrail(screen)
        text = understood.text
        if screen.blocked:
            log.info("input blocked reasons=%s", ",".join(screen.reasons))
            yield _counted(
                TurnResult(
                    response=_blocked_input_reply(screen),
                    state=state,
                    user_text=text or "[leere Nachricht]",
                )
            )
            return
        if screen.reasons:
            log.info("input redacted reasons=%s", ",".join(screen.reasons))
        # From here on only the sanitised message exists: answer model, transcript.
        sanitized = Message(role=Role.USER, content=text)
        conversation = conversation.model_copy(
            update={"messages": [*conversation.messages[:-1], sanitized]}
        )
        notice = not PAYMENT_DATA_REASONS.isdisjoint(screen.reasons)
        if notice:
            yield TextDelta(text=f"{PAYMENT_DATA_NOTICE}\n\n")
        async for event in self._turn(conversation, recovered, understood, session_id):
            if isinstance(event, TurnResult):
                if notice:
                    event = event.model_copy(
                        update={"response": with_payment_data_notice(event.response)}
                    )
                event = _counted(event)
            yield event

    async def _turn(
        self,
        conversation: Conversation,
        state: SessionState,
        understood: Understanding,
        session_id: str,
    ) -> AsyncIterator[TurnEvent]:
        """Stage 2: the turn graph. A provider outage in any step ends the turn with the contact
        channels and the previous state (NFR-11)."""
        text = conversation.messages[-1].content
        final: TurnState = {
            "conversation": conversation,
            "session": state,
            "routing": understood.routing,
            "verdict": understood.verdict,
        }
        context = TurnContext(session_id=session_id, tenant_id=self._tenant_id)
        parts = self._graph.astream(final, context=context, stream_mode=["custom", "values"])
        try:
            async for mode, data in cast(AsyncIterator[tuple[str, Any]], parts):
                if mode == "custom":
                    yield cast(TextDelta, data)
                else:
                    final = cast(TurnState, data)
        except ProviderUnavailableError as exc:
            log.error("provider unavailable: %s", exc)
            yield TurnResult(response=outage_contact_reply(), state=state, user_text=text)
            return
        yield TurnResult(response=final["response"], state=final["session"], user_text=text)


def recover_state(conversation: Conversation, state: SessionState) -> SessionState:
    """A stateless call (the evaluation) has the transcript but no session state. Rebuild what
    a session would hold from the earlier messages: the order last talked about (looked up again,
    scoped to the customer, before any use), an open ticket question of the assistant, and a
    ticket already opened. Sessions with state keep theirs."""
    if state != SessionState() or len(conversation.messages) < 2:
        return state
    earlier = conversation.messages[:-1]
    slots: dict[str, str] = {}
    for message in earlier:
        if order_id := extract_order_id(message.content):
            slots["order_id"] = order_id
        if message.role is Role.ASSISTANT and (match := _TICKET_NUMBER.search(message.content)):
            slots["ticket_id"] = match.group(1)
    last = earlier[-1]
    if last.role is not Role.ASSISTANT or TICKET_QUESTION not in last.content:
        return SessionState(slots=slots) if slots else state
    question = last.content[last.content.rfind("Soll ich") :]
    slots["question"] = question
    if order_id := extract_order_id(question):
        slots["order_id"] = order_id
    requests = [m.content for m in earlier[:-1] if m.role is Role.USER][-3:]
    if requests:
        slots["request"] = " / ".join(requests)
    return SessionState(
        active_intent=intent_of_question(question),
        workflow=WorkflowState.AWAITING_TICKET_CONFIRMATION,
        slots=slots,
    )


def _blocked_input_reply(screen: GuardrailDecision) -> AgentResponse:
    response = fallback(screen.reasons)
    if "prompt_injection" in screen.reasons:
        # Taxonomy: attempts to change the assistant's instructions are OUT_OF_SCOPE.
        return response.model_copy(update={"intent": Intent.OUT_OF_SCOPE})
    return response


def _counted(turn: TurnResult) -> TurnResult:
    response = turn.response
    AGENT_REPLIES.labels(behavior=response.behavior, intent=str(response.intent)).inc()
    return turn
