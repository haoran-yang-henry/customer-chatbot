"""Conversation orchestrator: the ``CustomerServiceAgent`` behind the API and the evaluation.

One turn: input guardrail → intent router ‖ attack classifier → state machine (``decide``) →
handler (knowledge answer with output guardrail, order guidance, ticket flow). Guardrails judge;
this class decides what the customer sees (ADR-0011). Tickets only for confirmed change
requests, at most one per conversation (ADR-0014).
"""

import asyncio
import re
from collections.abc import AsyncIterator

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
from customer_agent.application.guardrails import PAYMENT_DATA_REASONS, check_input
from customer_agent.application.guardrails.classifier import PromptAttackClassifier
from customer_agent.application.order_guidance import order_guidance, order_not_found
from customer_agent.application.ports import (
    AuditEntry,
    AuditLog,
    Clock,
    NewTicket,
    OrderRepository,
    ProviderUnavailableError,
    TicketService,
)
from customer_agent.application.routing import IntentRouter, route_by_rules
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
from customer_agent.observability import get_logger, record_guardrail, request_id_var
from customer_agent.observability import step as agent_step  # `step` is a Decision here
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
        router: IntentRouter,
        answers: AnswerService,
        *,
        orders: OrderRepository,
        tickets: TicketService,
        audit: AuditLog,
        clock: Clock,
        tenant_id: str,
        classifier: PromptAttackClassifier | None = None,
        max_tickets_per_day: int | None = 3,  # None: no limit
    ) -> None:
        self._router = router
        self._answers = answers
        self._orders = orders
        self._tickets = tickets
        self._audit = audit
        self._clock = clock
        self._tenant_id = tenant_id
        self._classifier = classifier
        self._max_tickets_per_day = max_tickets_per_day

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
        raw = conversation.messages[-1].content
        with agent_step("input_guardrail"):
            screen = check_input(raw)
        record_guardrail(screen)
        text = screen.content if screen.content is not None else raw
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
        # From here on only the sanitised message exists: router, answer model, transcript.
        sanitized = Message(role=Role.USER, content=text)
        conversation = conversation.model_copy(
            update={"messages": [*conversation.messages[:-1], sanitized]}
        )
        notice = not PAYMENT_DATA_REASONS.isdisjoint(screen.reasons)
        if notice:
            yield TextDelta(text=f"{PAYMENT_DATA_NOTICE}\n\n")
        async for event in self._turn(conversation, state, session_id):
            if isinstance(event, TurnResult):
                if notice:
                    event = event.model_copy(
                        update={"response": with_payment_data_notice(event.response)}
                    )
                event = _counted(event)
            yield event

    async def _turn(
        self, conversation: Conversation, state: SessionState, session_id: str
    ) -> AsyncIterator[TurnEvent]:
        text = conversation.messages[-1].content
        state = recover_ticket_state(conversation, state)
        try:
            routing, verdict = await self._route_and_screen(conversation, state, session_id)
            if verdict is not None:
                record_guardrail(verdict)
            if verdict is not None and verdict.blocked:
                log.warning("input blocked by classifier reasons=%s", ",".join(verdict.reasons))
                yield TurnResult(response=_vetoed(verdict, routing), state=state, user_text=text)
                return
            step = decide(state, routing, authenticated=conversation.customer_id is not None)
            log.info("step=%s intent=%s workflow=%s", step.step, step.intent, step.state.workflow)
            async for event in self._execute(conversation, routing, step, session_id):
                yield event
        except ProviderUnavailableError as exc:
            log.error("provider unavailable: %s", exc)
            yield TurnResult(
                # Nothing can be understood (NFR-11): contact channels at once, never a ticket.
                response=outage_contact_reply(),
                state=state,
                user_text=text,
            )

    async def _route_and_screen(
        self, conversation: Conversation, state: SessionState, session_id: str
    ) -> tuple[RoutingResult, GuardrailDecision | None]:
        """Router and classifier run concurrently: the classifier costs a call, not latency.
        Rule-routed messages (a greeting, a bare order number) skip the classifier."""
        if self._classifier is None or route_by_rules(conversation, state) is not None:
            return await self._router.route(conversation, state, session_id=session_id), None
        try:
            async with asyncio.TaskGroup() as group:
                routed = group.create_task(
                    self._router.route(conversation, state, session_id=session_id)
                )
                screened = group.create_task(
                    self._classifier.classify(conversation, session_id=session_id)
                )
        except ExceptionGroup as errors:  # keep ProviderUnavailableError catchable
            raise errors.exceptions[0] from None
        return routed.result(), screened.result()

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

        match step.step:
            case NextStep.ORDER_GUIDANCE:
                yield result(*await self._order_guidance(conversation, routing, step, session_id))
            case NextStep.ANSWER_FROM_KNOWLEDGE:
                async for event in self._answers.answer_stream(
                    conversation, query=routing.standalone_query, session_id=session_id
                ):
                    if isinstance(event, FinalAnswer):
                        yield result(event.response)
                    else:
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
        """Read-only, scoped to the authenticated customer in SQL; every lookup is audited."""
        customer_id = conversation.customer_id
        if customer_id is None:  # decide() never routes anonymous users here
            raise RuntimeError("order lookups require an authenticated customer")
        with agent_step("tool.get_order_status"):
            order = await self._orders.get_for_customer(
                order_id, customer_id=customer_id, tenant_id=self._tenant_id
            )
        await self._audit.record(
            AuditEntry(
                tool="get_order_status",
                customer_id=customer_id,
                session_id=session_id,
                request_id=request_id_var.get(),
                arguments={"order_id": order_id},
                outcome="found" if order else "not_found",
            )
        )
        return order

    async def _order_guidance(
        self,
        conversation: Conversation,
        routing: RoutingResult,
        step: Decision,
        session_id: str,
    ) -> tuple[AgentResponse, SessionState]:
        """Facts and eligibility for one own order; if the requested change is still possible,
        one question whether to forward it (or a pointer to the ticket already open)."""
        state = step.state
        order_id = state.slots["order_id"]
        intent = step.intent or Intent.ORDER_STATUS
        order = await self._lookup(conversation, order_id, session_id)
        if order is None:
            # Never remember (or later attach to a ticket) an order that is not the customer's.
            return order_not_found(order_id, intent), _forget_order(state)
        guidance = order_guidance(intent, order, self._clock.today())
        response = guidance.response
        if not guidance.change_possible:
            return response, state
        if "ticket_id" in state.slots:
            reminder = ticket_exists_reply(state.slots["ticket_id"], intent=intent).answer
            return response.model_copy(update={"answer": f"{response.answer}\n\n{reminder}"}), state
        question = ticket_question(intent, order_id)
        pending = state.model_copy(
            update={
                "workflow": WorkflowState.AWAITING_TICKET_CONFIRMATION,
                "slots": {**state.slots, "request": routing.standalone_query},
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
        return confirm_ticket_reply(step.intent, order_id), state

    async def _create_ticket(
        self, conversation: Conversation, step: Decision, session_id: str
    ) -> tuple[AgentResponse, SessionState]:
        state = step.state
        ticket_id, created = await self._open_ticket(
            conversation,
            session_id,
            intent=step.intent,
            order_id=state.slots.get("order_id"),
            request=state.slots.get("request"),
        )
        slots = {k: v for k, v in state.slots.items() if k != "request"}
        done = state.model_copy(update={"slots": {**slots, "ticket_id": ticket_id}})
        if not created:
            return ticket_exists_reply(ticket_id, intent=step.intent), done
        return ticket_created_reply(ticket_id, intent=step.intent), done

    async def _open_ticket(
        self,
        conversation: Conversation,
        session_id: str,
        *,
        intent: Intent | None,
        order_id: str | None,
        request: str | None,
    ) -> tuple[str, bool]:
        """``(ticket number, created)``; at the limit the newest open ticket, not a new one."""
        customer_id = conversation.customer_id
        arguments = {"order_id": order_id or "-", "intent": str(intent)}

        async def audit(outcome: str) -> None:
            await self._audit.record(
                AuditEntry(
                    tool="create_ticket",
                    customer_id=customer_id,
                    session_id=session_id,
                    request_id=request_id_var.get(),
                    arguments=arguments,
                    outcome=outcome,
                )
            )

        if customer_id is not None and self._max_tickets_per_day is not None:
            recent = await self._tickets.recent(customer_id=customer_id, tenant_id=self._tenant_id)
            if len(recent) >= self._max_tickets_per_day:
                # Rate limit on the only state-changing tool (T12): point to the open ticket.
                await audit(f"rate_limited:{recent[0]}")
                log.warning("ticket limit reached recent=%d", len(recent))
                return recent[0], False
        # The confirmed request, not the confirmation itself ("Ja, bitte").
        summary = request or conversation.messages[-1].content
        with agent_step("tool.create_ticket"):
            ticket_id = await self._tickets.create(
                NewTicket(
                    tenant_id=self._tenant_id,
                    customer_id=customer_id,
                    session_id=session_id,
                    intent=intent.value if intent else None,
                    order_id=order_id,
                    summary=f"{intent or 'UNKNOWN'}: {summary[:500]}",
                )
            )
        await audit(ticket_id)
        log.info("ticket=%s intent=%s order_id=%s", ticket_id, intent, order_id)
        return ticket_id, True


def recover_ticket_state(conversation: Conversation, state: SessionState) -> SessionState:
    """A stateless call (the evaluation) has the transcript but no session state. Rebuild what
    the ticket flow needs from the assistant's own earlier messages: an open confirmation
    question and a ticket already opened. Sessions with state keep theirs."""
    if state != SessionState():
        return state
    messages = conversation.messages
    slots: dict[str, str] = {}
    for message in messages[:-1]:
        if message.role is Role.ASSISTANT and (match := _TICKET_NUMBER.search(message.content)):
            slots["ticket_id"] = match.group(1)
    last = messages[-2] if len(messages) >= 2 else None
    if last is None or last.role is not Role.ASSISTANT or TICKET_QUESTION not in last.content:
        return SessionState(slots=slots) if slots else state
    question = last.content[last.content.rfind("Soll ich") :]
    if order_id := extract_order_id(question):
        slots["order_id"] = order_id
    earlier = [m.content for m in messages[:-2] if m.role is Role.USER][-3:]
    if earlier:
        slots["request"] = " / ".join(earlier)
    return SessionState(
        active_intent=intent_of_question(question),
        workflow=WorkflowState.AWAITING_TICKET_CONFIRMATION,
        slots=slots,
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
