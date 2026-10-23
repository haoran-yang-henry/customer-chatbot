"""Conversation orchestrator: the ``CustomerServiceAgent`` behind the API and the evaluation.

One turn: understand agent (input guardrail, rule routing, intent router ‖ attack classifier) →
state machine (``decide``) → handler (knowledge answer with output guardrail, order guidance,
ticket flow). Guardrails judge; this class decides what the customer sees (ADR-0011). Tickets
only for confirmed change requests, at most one per conversation (ADR-0014).
"""

import re
from collections.abc import AsyncIterator
from typing import cast

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, ConfigDict

from customer_agent.application.answering import (
    AnswerService,
    FinalAnswer,
    TextDelta,
    stateless_session_id,
)
from customer_agent.application.fallbacks import (
    PAYMENT_DATA_NOTICE,
    TICKET_QUESTION,
    confirm_ticket_reply,
    contact_reply,
    fallback,
    intent_of_question,
    outage_contact_reply,
    step_reply,
    ticket_created_reply,
    ticket_exists_reply,
    ticket_question,
    with_payment_data_notice,
)
from customer_agent.application.guardrails import PAYMENT_DATA_REASONS
from customer_agent.application.order_guidance import order_guidance, order_not_found
from customer_agent.application.ports import (
    AuditLog,
    Clock,
    OrderRepository,
    ProviderUnavailableError,
    TicketService,
)
from customer_agent.application.tools import (
    CreateTicket,
    GetOrderStatus,
    TicketOutcome,
    tool_config,
)
from customer_agent.application.understanding import UnderstandAgent, Understanding
from customer_agent.domain.conversation import AgentResponse, Conversation, Message, Role
from customer_agent.domain.guardrails import GuardrailDecision
from customer_agent.domain.intents import Intent
from customer_agent.domain.orders import Order
from customer_agent.domain.routing import RoutingResult, extract_order_id
from customer_agent.domain.session import (
    Decision,
    NextStep,
    SessionState,
    WorkflowState,
    decide,
)
from customer_agent.observability import get_logger, record_guardrail
from customer_agent.observability import step as agent_step  # `step` is a Decision here
from customer_agent.observability.metrics import AGENT_REPLIES

log = get_logger("chat")

_TICKET_NUMBER = re.compile(r"Ticketnummer (?:lautet )?(T-\d{6})")
# Facts the customer may simply ask for again ("Und wo ist sie jetzt?"); everything else is
# guidance, which is not repeated word for word.
_FACT_LOOKUPS = {Intent.ORDER_STATUS, Intent.REFUND_STATUS}


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
        self._answers = answers
        self._order_tool = GetOrderStatus(orders=orders, audit_log=audit)
        self._ticket_tool = CreateTicket(
            tickets=tickets, audit_log=audit, max_per_day=max_tickets_per_day
        )
        self._clock = clock
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
        text = conversation.messages[-1].content
        routing, verdict = understood.routing, understood.verdict
        if routing is None:
            # Nothing was understood (NFR-11): contact channels at once, never a ticket.
            yield TurnResult(response=outage_contact_reply(), state=state, user_text=text)
            return
        if verdict is not None:
            record_guardrail(verdict)
            if verdict.blocked:
                log.warning("input blocked by classifier reasons=%s", ",".join(verdict.reasons))
                yield TurnResult(response=_vetoed(verdict, routing), state=state, user_text=text)
                return
        try:
            step = decide(state, routing, authenticated=conversation.customer_id is not None)
            log.info("step=%s intent=%s workflow=%s", step.step, step.intent, step.state.workflow)
            async for event in self._execute(conversation, routing, step, session_id):
                yield event
        except ProviderUnavailableError as exc:
            log.error("provider unavailable: %s", exc)
            yield TurnResult(response=outage_contact_reply(), state=state, user_text=text)

    async def _execute(
        self,
        conversation: Conversation,
        routing: RoutingResult,
        step: Decision,
        session_id: str,
    ) -> AsyncIterator[TurnEvent]:
        def result(response: AgentResponse, state: SessionState | None = None) -> TurnResult:
            final = response.model_copy(update={"intent": step.intent})
            log.info("behavior=%s sources=%s", final.behavior, ",".join(final.sources) or "-")
            return TurnResult(
                response=final,
                state=state or step.state,
                user_text=conversation.messages[-1].content,
            )

        # A follow-up while a ticket question is open: answer it, then ask the question again.
        pending = step.state.slots.get("question") if _awaiting(step.state) else None

        async def knowledge() -> AsyncIterator[TurnEvent]:
            async for event in self._answers.answer_stream(
                conversation, query=routing.standalone_query, session_id=session_id
            ):
                if not isinstance(event, FinalAnswer):
                    yield event
                elif pending is None:
                    yield result(event.response)
                else:
                    answer = f"{event.response.answer}\n\n{pending}"
                    yield result(event.response.model_copy(update={"answer": answer}))

        match step.step:
            case NextStep.ORDER_GUIDANCE:
                reply = await self._order_guidance(conversation, routing, step, session_id)
                if reply is not None:
                    yield result(*reply)
                else:
                    # The same guidance once more would not help: the customer asks something
                    # the template does not cover, so answer the follow-up from the documents.
                    async for event in knowledge():
                        yield event
            case NextStep.ANSWER_FROM_KNOWLEDGE:
                async for event in knowledge():
                    yield event
            case NextStep.CONFIRM_TICKET:
                yield result(*await self._confirm_ticket(conversation, step, session_id))
            case NextStep.CREATE_TICKET:
                yield result(*await self._create_ticket(conversation, step, session_id))
            case NextStep.TICKET_EXISTS:
                yield result(ticket_exists_reply(step.state.slots["ticket_id"], intent=step.intent))
            case NextStep.CONTACT:
                yield result(
                    contact_reply(intent=step.intent, change_request=routing.change_request)
                )
            case NextStep.CLARIFY:
                yield result(step_reply("clarify", text=routing.clarification_question))
            case _:
                yield result(step_reply(step.step.value))

    async def _lookup(
        self, conversation: Conversation, order_id: str, session_id: str
    ) -> Order | None:
        order = await self._order_tool.ainvoke(
            {"order_id": order_id}, config=self._tool_config(conversation, session_id)
        )
        return cast(Order | None, order)

    def _tool_config(self, conversation: Conversation, session_id: str) -> RunnableConfig:
        return tool_config(
            customer_id=conversation.customer_id, tenant_id=self._tenant_id, session_id=session_id
        )

    async def _order_guidance(
        self,
        conversation: Conversation,
        routing: RoutingResult,
        step: Decision,
        session_id: str,
    ) -> tuple[AgentResponse, SessionState] | None:
        """Facts and eligibility for one own order; if the requested change is still possible,
        one question whether to forward it (or a pointer to the ticket already open). ``None``
        if this guidance was already given in the conversation (see ``_execute``)."""
        state = step.state
        order_id = state.slots["order_id"]
        intent = step.intent or Intent.ORDER_STATUS
        order = await self._lookup(conversation, order_id, session_id)
        if order is None:
            # Never remember (or later attach to a ticket) an order that is not the customer's.
            return order_not_found(order_id, intent), _forget_order(state)
        guidance = order_guidance(
            intent, order, self._clock.today(), defect=routing.defect_reported
        )
        response = guidance.response
        if intent not in _FACT_LOOKUPS and _already_said(conversation, response.answer):
            return None
        if not guidance.change_possible:
            return response, state
        if "ticket_id" in state.slots:
            reminder = ticket_exists_reply(state.slots["ticket_id"], intent=intent).answer
            return response.model_copy(update={"answer": f"{response.answer}\n\n{reminder}"}), state
        question = ticket_question(intent, order_id, subject=guidance.question_subject)
        pending = state.model_copy(
            update={
                "workflow": WorkflowState.AWAITING_TICKET_CONFIRMATION,
                "slots": {**state.slots, "request": routing.standalone_query, "question": question},
            }
        )
        return response.model_copy(update={"answer": f"{response.answer}\n\n{question}"}), pending

    async def _confirm_ticket(
        self, conversation: Conversation, step: Decision, session_id: str
    ) -> tuple[AgentResponse, SessionState]:
        """Ask once; an order the request names must be the customer's before it is mentioned."""
        state = step.state
        order_id = state.slots.get("order_id")
        if order_id is not None and await self._lookup(conversation, order_id, session_id) is None:
            slots = {k: v for k, v in state.slots.items() if k not in ("order_id", "request")}
            idle = state.model_copy(update={"workflow": WorkflowState.IDLE, "slots": slots})
            return order_not_found(order_id, step.intent or Intent.HUMAN_AGENT), idle
        question = ticket_question(step.intent, order_id)
        asked = state.model_copy(update={"slots": {**state.slots, "question": question}})
        return confirm_ticket_reply(step.intent, order_id), asked

    async def _create_ticket(
        self, conversation: Conversation, step: Decision, session_id: str
    ) -> tuple[AgentResponse, SessionState]:
        state = step.state
        arguments = {
            "intent": step.intent,
            "order_id": state.slots.get("order_id"),
            # The confirmed request, not the confirmation itself ("Ja, bitte").
            "request": state.slots.get("request") or conversation.messages[-1].content,
        }
        outcome = cast(
            TicketOutcome,
            await self._ticket_tool.ainvoke(
                arguments, config=self._tool_config(conversation, session_id)
            ),
        )
        slots = {k: v for k, v in state.slots.items() if k != "request"}
        done = state.model_copy(update={"slots": {**slots, "ticket_id": outcome.ticket_id}})
        if not outcome.created:  # the daily limit: the newest open ticket, not a new one
            return ticket_exists_reply(outcome.ticket_id, intent=step.intent), done
        return ticket_created_reply(outcome.ticket_id, intent=step.intent), done


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


def _awaiting(state: SessionState) -> bool:
    return state.workflow is WorkflowState.AWAITING_TICKET_CONFIRMATION


def _already_said(conversation: Conversation, text: str) -> bool:
    return any(
        text in message.content
        for message in conversation.messages[:-1]
        if message.role is Role.ASSISTANT
    )


def _forget_order(state: SessionState) -> SessionState:
    return state.model_copy(
        update={"slots": {k: v for k, v in state.slots.items() if k != "order_id"}}
    )


def _blocked_input_reply(screen: GuardrailDecision) -> AgentResponse:
    response = fallback(screen.reasons)
    if "prompt_injection" in screen.reasons:
        # Taxonomy: attempts to change the assistant's instructions are OUT_OF_SCOPE.
        return response.model_copy(update={"intent": Intent.OUT_OF_SCOPE})
    return response


def _vetoed(verdict: GuardrailDecision, routing: RoutingResult) -> AgentResponse:
    """The classifier vetoes the step (no lookup, no ticket, no answer); the most specific
    refusal the router's flags allow still applies."""
    if routing.third_party:
        response = step_reply("refuse_third_party")
    elif routing.money_demand:
        response = step_reply("refuse_money_demand")
    else:
        response = fallback(verdict.reasons)
    return response.model_copy(update={"intent": routing.intent, "reasons": verdict.reasons})


def _counted(turn: TurnResult) -> TurnResult:
    response = turn.response
    AGENT_REPLIES.labels(behavior=response.behavior, intent=str(response.intent)).inc()
    return turn
