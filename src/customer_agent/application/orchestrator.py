"""Conversation orchestrator: the ``CustomerServiceAgent`` behind the API and the evaluation.

One turn: input guardrail → intent router → state machine (``decide``) → handler. Order
guidance with real order data (Day 5) and full guardrails (Day 6) plug in here.
"""

from collections.abc import AsyncIterator

from pydantic import BaseModel, ConfigDict

from customer_agent.application.answering import (
    AnswerService,
    FinalAnswer,
    TextDelta,
    stateless_session_id,
)
from customer_agent.application.fallbacks import contact_reply, fallback, handoff_reply, step_reply
from customer_agent.application.guardrails import check_input
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
from customer_agent.application.routing import IntentRouter
from customer_agent.domain.conversation import AgentResponse, Conversation
from customer_agent.domain.intents import Intent
from customer_agent.domain.routing import RoutingResult
from customer_agent.domain.session import Decision, NextStep, SessionState, decide
from customer_agent.observability import get_logger, request_id_var

log = get_logger("chat")


class TurnResult(BaseModel):
    """The agent's reply plus the session state to persist for the next turn."""

    model_config = ConfigDict(frozen=True)

    response: AgentResponse
    state: SessionState


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
    ) -> None:
        self._router = router
        self._answers = answers
        self._orders = orders
        self._tickets = tickets
        self._audit = audit
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
        decision = check_input(conversation.messages[-1].content)
        if decision.blocked:
            log.info("input blocked reasons=%s", ",".join(decision.reasons))
            yield TurnResult(response=fallback(decision.reasons), state=state)
            return
        try:
            routing = await self._router.route(conversation, state, session_id=session_id)
            step = decide(state, routing, authenticated=conversation.customer_id is not None)
            log.info("step=%s intent=%s workflow=%s", step.step, step.intent, step.state.workflow)
            async for event in self._execute(conversation, routing, step, session_id):
                yield event
        except ProviderUnavailableError as exc:
            log.error("provider unavailable: %s", exc)
            yield TurnResult(response=fallback(["provider_unavailable"]), state=state)

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
            return TurnResult(response=final, state=step.state)

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
        message = conversation.messages[-1].content
        ticket_id = await self._tickets.create(
            NewTicket(
                tenant_id=self._tenant_id,
                customer_id=conversation.customer_id,
                session_id=session_id,
                intent=step.intent.value if step.intent else None,
                order_id=order_id,
                summary=f"{step.intent or 'UNKNOWN'}: {message[:500]}",
            )
        )
        await self._audit.record(
            AuditEntry(
                tool="create_ticket",
                customer_id=conversation.customer_id,
                session_id=session_id,
                request_id=request_id_var.get(),
                arguments={"order_id": order_id or "-", "intent": str(step.intent)},
                outcome=ticket_id,
            )
        )
        log.info("handoff ticket=%s intent=%s order_id=%s", ticket_id, step.intent, order_id)
        return handoff_reply(ticket_id, intent=step.intent)
