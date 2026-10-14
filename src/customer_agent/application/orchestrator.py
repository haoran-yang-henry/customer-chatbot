"""Conversation orchestrator: the ``CustomerServiceAgent`` behind the API and the evaluation.

One turn: input guardrail → intent router ‖ attack classifier → state machine (``decide``) →
handler (knowledge answer with output guardrail, order guidance, handoff). Guardrails judge;
this class decides what the customer sees (ADR-0011).
"""

import asyncio
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
    contact_reply,
    fallback,
    handoff_pending_reply,
    handoff_reply,
    outage_contact_reply,
    outage_reply,
    step_reply,
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
from customer_agent.domain.routing import RoutingResult
from customer_agent.domain.session import Decision, NextStep, SessionState, decide
from customer_agent.observability import get_logger, record_guardrail, request_id_var
from customer_agent.observability import step as agent_step  # `step` is a Decision here
from customer_agent.observability.metrics import AGENT_REPLIES

log = get_logger("chat")


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
                response=await self._outage_reply(conversation, session_id),
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
        def result(response: AgentResponse) -> TurnResult:
            final = response.model_copy(update={"intent": step.intent})
            log.info("behavior=%s sources=%s", final.behavior, ",".join(final.sources) or "-")
            return TurnResult(
                response=final, state=step.state, user_text=conversation.messages[-1].content
            )

        match step.step:
            case NextStep.ORDER_GUIDANCE:
                yield result(await self._order_guidance(conversation, step, session_id))
            case NextStep.ANSWER_FROM_KNOWLEDGE:
                async for event in self._answers.answer_stream(
                    conversation, query=routing.standalone_query, session_id=session_id
                ):
                    if isinstance(event, FinalAnswer):
                        yield result(event.response)
                    else:
                        yield event
            case NextStep.HANDOFF:
                yield result(await self._handoff(conversation, routing, step, session_id))
            case NextStep.HANDOFF_CONTACT:
                yield result(contact_reply(intent=step.intent))
            case NextStep.CLARIFY:
                yield result(step_reply("clarify", text=routing.clarification_question))
            case _:
                yield result(step_reply(step.step.value))

    async def _order_guidance(
        self, conversation: Conversation, step: Decision, session_id: str
    ) -> AgentResponse:
        """Read-only lookup scoped to the authenticated customer, then deterministic guidance."""
        customer_id = conversation.customer_id
        order_id = step.state.slots["order_id"]
        intent = step.intent or Intent.ORDER_STATUS
        if customer_id is None:  # decide() never routes anonymous users here
            raise RuntimeError("order guidance requires an authenticated customer")
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
        if order is None:
            return order_not_found(order_id, intent)
        return order_guidance(intent, order, self._clock.today())

    async def _handoff(
        self,
        conversation: Conversation,
        routing: RoutingResult,
        step: Decision,
        session_id: str,
    ) -> AgentResponse:
        order_id = routing.order_id or step.state.slots.get("order_id")
        ticket_id, created = await self._open_ticket(
            conversation, session_id, intent=step.intent, order_id=order_id
        )
        if not created:
            return handoff_pending_reply(ticket_id, intent=step.intent)
        return handoff_reply(ticket_id, intent=step.intent)

    async def _outage_reply(self, conversation: Conversation, session_id: str) -> AgentResponse:
        """Nobody can answer automatically (NFR-11), so a colleague takes over."""
        if conversation.customer_id is None:
            return outage_contact_reply()
        try:
            ticket_id, created = await self._open_ticket(
                conversation, session_id, intent=None, order_id=None
            )
        except Exception:  # the database may be down as well; still answer without a 5xx
            log.exception("outage ticket could not be created")
            return fallback(["provider_unavailable"])
        return outage_reply(ticket_id, created=created)

    async def _open_ticket(
        self,
        conversation: Conversation,
        session_id: str,
        *,
        intent: Intent | None,
        order_id: str | None,
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
        message = conversation.messages[-1].content
        with agent_step("tool.create_ticket"):
            ticket_id = await self._tickets.create(
                NewTicket(
                    tenant_id=self._tenant_id,
                    customer_id=customer_id,
                    session_id=session_id,
                    intent=intent.value if intent else None,
                    order_id=order_id,
                    summary=f"{intent or 'UNKNOWN'}: {message[:500]}",
                )
            )
        await audit(ticket_id)
        log.info("ticket=%s intent=%s order_id=%s", ticket_id, intent, order_id)
        return ticket_id, True


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
