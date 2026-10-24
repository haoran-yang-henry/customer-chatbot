"""Stage 2 of a turn (rebuild plan §4.2): what to do with an understood message, as a LangGraph.

    START → gate ─ blocked, outage or veto ─────────────────────────────────────→ finish → END
              └→ decide ─ ORDER_GUIDANCE ────→ order_guidance ─ already given → knowledge
                        ├ ANSWER_FROM_KNOWLEDGE → knowledge ──────────────────────→ finish
                        ├ CONFIRM_TICKET ─────→ confirm_ticket ───────────────────→ finish
                        ├ CREATE_TICKET ──────→ create_ticket ────────────────────→ finish
                        └ every other step ───→ template_reply ───────────────────→ finish

The state is the session: the masked transcript (``messages``, appended per turn) and the
``SessionState``; with a checkpointer (thread = session id) it outlives the turn. Every turn's
input sets the per-turn fields anew (input check, routing, verdict, decision, response). All of
it is masked: stage 1 checked the message before the graph sees it (invariant I3).

``decide()`` stays the pure state machine of ``domain/session.py``; the graph only routes on its
``Decision``. Guardrails judge; the nodes decide what the customer sees (ADR-0011). Tickets only
for confirmed change requests, at most one per conversation (ADR-0014). The answer's tokens and
the payment-data notice leave through the ``custom`` stream.
"""

import operator
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated, Any, NotRequired, Protocol, TypedDict, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from customer_agent.application.answering import AnswerService, FinalAnswer, TextDelta
from customer_agent.application.fallbacks import (
    PAYMENT_DATA_NOTICE,
    confirm_ticket_reply,
    contact_reply,
    fallback,
    outage_contact_reply,
    step_reply,
    ticket_created_reply,
    ticket_exists_reply,
    ticket_question,
    with_payment_data_notice,
)
from customer_agent.application.feedback import FeedbackPolicy
from customer_agent.application.guardrails import PAYMENT_DATA_REASONS
from customer_agent.application.order_guidance import order_guidance, order_not_found
from customer_agent.application.ports import Clock, ProviderUnavailableError
from customer_agent.application.tools import (
    CreateTicket,
    GetOrderStatus,
    TicketOutcome,
    tool_config,
)
from customer_agent.domain.conversation import AgentResponse, Conversation, Message, Role
from customer_agent.domain.guardrails import GuardrailDecision
from customer_agent.domain.intents import Intent
from customer_agent.domain.orders import Order
from customer_agent.domain.routing import RoutingResult
from customer_agent.domain.session import Decision, NextStep, SessionState, WorkflowState, decide
from customer_agent.observability import get_logger, record_guardrail
from customer_agent.observability.metrics import AGENT_REPLIES

log = get_logger("chat")

# Transcript passed to the router and the answer prompt; older turns stay in the checkpoint.
MAX_CONTEXT_MESSAGES = 20
# Facts the customer may simply ask for again ("Und wo ist sie jetzt?"); everything else is
# guidance, which is not repeated word for word.
_FACT_LOOKUPS = {Intent.ORDER_STATUS, Intent.REFUND_STATUS}
# Everything the state holds; a checkpoint may deserialize these and nothing else.
CHECKPOINT_TYPES = [
    ("customer_agent.domain.conversation", "AgentResponse"),
    ("customer_agent.domain.conversation", "Behavior"),
    ("customer_agent.domain.conversation", "Message"),
    ("customer_agent.domain.conversation", "Role"),
    ("customer_agent.domain.guardrails", "GuardrailAction"),
    ("customer_agent.domain.guardrails", "GuardrailDecision"),
    ("customer_agent.domain.intents", "Intent"),
    ("customer_agent.domain.routing", "RoutingResult"),
    ("customer_agent.domain.routing", "RoutingSource"),
    ("customer_agent.domain.session", "Decision"),
    ("customer_agent.domain.session", "NextStep"),
    ("customer_agent.domain.session", "SessionState"),
    ("customer_agent.domain.session", "WorkflowState"),
]


class TurnState(TypedDict):
    messages: Annotated[list[Message], operator.add]  # masked transcript
    session: SessionState
    input: GuardrailDecision  # this turn's input check; its content is the masked message
    routing: RoutingResult | None  # None: blocked, or the router's provider failed (NFR-11)
    verdict: GuardrailDecision | None
    decision: Decision | None
    response: AgentResponse | None
    feedback_requested: bool  # this reply asks whether the conversation helped
    feedback_asked: NotRequired[bool]  # the session was asked once: never again


def turn_input(
    messages: list[Message],
    session: SessionState,
    *,
    screen: GuardrailDecision,
    routing: RoutingResult | None,
    verdict: GuardrailDecision | None,
) -> TurnState:
    """``messages``: what to append, ending with this turn's masked message (with a
    checkpointer only that message; without one the whole conversation)."""
    return {
        "messages": messages,
        "session": session,
        "input": screen,
        "routing": routing,
        "verdict": verdict,
        "decision": None,
        "response": None,
        "feedback_requested": False,
    }


@dataclass(frozen=True)
class TurnContext:
    session_id: str
    tenant_id: str
    customer_id: str | None


Update = dict[str, Any]


class _Step(Protocol):
    def __call__(self, state: TurnState, *, runtime: Runtime[TurnContext]) -> Awaitable[Update]: ...


class TurnGraph:
    def __init__(
        self,
        answers: AnswerService,
        order_tool: GetOrderStatus,
        ticket_tool: CreateTicket,
        clock: Clock,
        feedback: FeedbackPolicy,
    ) -> None:
        self._answers = answers
        self._order_tool = order_tool
        self._ticket_tool = ticket_tool
        self._clock = clock
        self._feedback = feedback

    def compile(
        self, checkpointer: BaseCheckpointSaver[Any] | None = None
    ) -> CompiledStateGraph[TurnState, TurnContext, TurnState, TurnState]:
        graph = StateGraph(TurnState, context_schema=TurnContext)
        graph.add_node("gate", self._gate)
        graph.add_node("decide", self._decide)
        graph.add_node("order_guidance", _outage_safe(self._order_guidance))
        graph.add_node("knowledge", _outage_safe(self._knowledge))
        graph.add_node("confirm_ticket", _outage_safe(self._confirm_ticket))
        graph.add_node("create_ticket", _outage_safe(self._create_ticket))
        graph.add_node("template_reply", self._template_reply)
        graph.add_node("finish", self._finish)
        graph.add_edge(START, "gate")
        graph.add_conditional_edges("gate", _answered_or("decide"), ["decide", "finish"])
        graph.add_conditional_edges(
            "decide",
            _route_step,
            ["order_guidance", "knowledge", "confirm_ticket", "create_ticket", "template_reply"],
        )
        graph.add_conditional_edges(
            "order_guidance", _answered_or("knowledge"), ["knowledge", "finish"]
        )
        for node in ("knowledge", "confirm_ticket", "create_ticket", "template_reply"):
            graph.add_edge(node, "finish")
        graph.add_edge("finish", END)
        return graph.compile(checkpointer=checkpointer, name="turn")

    # --- nodes -------------------------------------------------------------------------------

    def _gate(self, state: TurnState, runtime: Runtime[TurnContext]) -> Update:
        """Blocked input, nothing understood, or the classifier's veto: a reply without a step."""
        screen = state["input"]
        if screen.blocked:
            log.info("input blocked reasons=%s", ",".join(screen.reasons))
            return {"response": _blocked_input_reply(screen)}
        if screen.reasons:
            log.info("input redacted reasons=%s", ",".join(screen.reasons))
        if _payment_data(screen):
            runtime.stream_writer(TextDelta(text=f"{PAYMENT_DATA_NOTICE}\n\n"))
        routing, verdict = state["routing"], state["verdict"]
        if routing is None:
            # Nothing was understood (NFR-11): contact channels at once, never a ticket.
            return {"response": outage_contact_reply()}
        if verdict is not None:
            record_guardrail(verdict)
            if verdict.blocked:
                log.warning("input blocked by classifier reasons=%s", ",".join(verdict.reasons))
                return {"response": _vetoed(verdict, routing)}
        return {}

    def _decide(self, state: TurnState, runtime: Runtime[TurnContext]) -> Update:
        authenticated = runtime.context.customer_id is not None
        decision = decide(state["session"], _routing(state), authenticated=authenticated)
        log.info(
            "step=%s intent=%s workflow=%s",
            decision.step,
            decision.intent,
            decision.state.workflow,
        )
        return {"decision": decision}

    async def _order_guidance(self, state: TurnState, runtime: Runtime[TurnContext]) -> Update:
        """Facts and eligibility for one own order; if the requested change is still possible,
        one question whether to forward it (or a pointer to the ticket already open). Nothing if
        this guidance was already given: the knowledge node answers the follow-up instead."""
        decision = _decision(state)
        conversation = _conversation(state, runtime)
        session = decision.state
        order_id = session.slots["order_id"]
        intent = decision.intent or Intent.ORDER_STATUS
        order = await self._lookup(order_id, runtime.context)
        if order is None:
            # Never remember (or later attach to a ticket) an order that is not the customer's.
            return _done(state, order_not_found(order_id, intent), _forget_order(session))
        guidance = order_guidance(
            intent, order, self._clock.today(), defect=_routing(state).defect_reported
        )
        response = guidance.response
        if intent not in _FACT_LOOKUPS and _already_said(conversation, response.answer):
            return {}
        if not guidance.change_possible:
            return _done(state, response)
        if "ticket_id" in session.slots:
            reminder = ticket_exists_reply(session.slots["ticket_id"], intent=intent).answer
            return _done(state, _appended(response, reminder))
        question = ticket_question(intent, order_id, subject=guidance.question_subject)
        pending = session.model_copy(
            update={
                "workflow": WorkflowState.AWAITING_TICKET_CONFIRMATION,
                "slots": {
                    **session.slots,
                    "request": _routing(state).standalone_query,
                    "question": question,
                },
            }
        )
        return _done(state, _appended(response, question), pending)

    async def _knowledge(self, state: TurnState, runtime: Runtime[TurnContext]) -> Update:
        """The grounded answer; a ticket question still open is asked again after it."""
        decision = _decision(state)
        pending = decision.state.slots.get("question") if _awaiting(decision.state) else None
        response: AgentResponse | None = None
        async for event in self._answers.answer_stream(
            _conversation(state, runtime),
            query=_routing(state).standalone_query,
            session_id=runtime.context.session_id,
        ):
            if isinstance(event, FinalAnswer):
                response = event.response
            else:
                runtime.stream_writer(event)
        if response is None:  # pragma: no cover - answer_stream always ends with FinalAnswer
            raise RuntimeError("answer ended without a final answer")
        return _done(state, _appended(response, pending) if pending else response)

    async def _confirm_ticket(self, state: TurnState, runtime: Runtime[TurnContext]) -> Update:
        """Ask once; an order the request names must be the customer's before it is mentioned."""
        decision = _decision(state)
        session = decision.state
        order_id = session.slots.get("order_id")
        if order_id is not None and await self._lookup(order_id, runtime.context) is None:
            slots = {k: v for k, v in session.slots.items() if k not in ("order_id", "request")}
            idle = session.model_copy(update={"workflow": WorkflowState.IDLE, "slots": slots})
            intent = decision.intent or Intent.HUMAN_AGENT
            return _done(state, order_not_found(order_id, intent), idle)
        question = ticket_question(decision.intent, order_id)
        asked = session.model_copy(update={"slots": {**session.slots, "question": question}})
        return _done(state, confirm_ticket_reply(decision.intent, order_id), asked)

    async def _create_ticket(self, state: TurnState, runtime: Runtime[TurnContext]) -> Update:
        decision = _decision(state)
        session = decision.state
        arguments = {
            "intent": decision.intent,
            "order_id": session.slots.get("order_id"),
            # The confirmed request, not the confirmation itself ("Ja, bitte").
            "request": session.slots.get("request") or state["messages"][-1].content,
        }
        outcome = cast(
            TicketOutcome,
            await self._ticket_tool.ainvoke(arguments, config=_tool_config(runtime.context)),
        )
        slots = {k: v for k, v in session.slots.items() if k != "request"}
        done = session.model_copy(update={"slots": {**slots, "ticket_id": outcome.ticket_id}})
        # At the daily limit the tool returns the newest open ticket instead of a new one.
        reply = ticket_created_reply if outcome.created else ticket_exists_reply
        return _done(state, reply(outcome.ticket_id, intent=decision.intent), done)

    def _template_reply(self, state: TurnState) -> Update:
        decision = _decision(state)
        match decision.step:
            case NextStep.TICKET_EXISTS:
                ticket_id = decision.state.slots["ticket_id"]
                response = ticket_exists_reply(ticket_id, intent=decision.intent)
            case NextStep.CONTACT:
                change = _routing(state).change_request
                response = contact_reply(intent=decision.intent, change_request=change)
            case NextStep.CLARIFY:
                response = step_reply("clarify", text=_routing(state).clarification_question)
            case _:
                response = step_reply(decision.step.value)
        return _done(state, response)

    def _finish(self, state: TurnState, runtime: Runtime[TurnContext]) -> Update:
        """The reply joins the transcript as the customer sees it; a sampled session of several
        rounds is asked once whether the conversation helped (``FeedbackPolicy``)."""
        response = state["response"]
        if response is None:  # pragma: no cover - every path to finish sets a response
            raise RuntimeError("turn ended without a response")
        if _payment_data(state["input"]):
            response = with_payment_data_notice(response)
        AGENT_REPLIES.labels(behavior=response.behavior, intent=str(response.intent)).inc()
        reply = Message(role=Role.ASSISTANT, content=response.answer)
        update: Update = {"response": response, "messages": [reply]}
        rounds = sum(1 for message in state["messages"] if message.role is Role.USER)
        if not state.get("feedback_asked", False) and self._feedback.asks(
            runtime.context.session_id, rounds, response.behavior
        ):
            log.info("feedback requested rounds=%d", rounds)
            update |= {"feedback_requested": True, "feedback_asked": True}
        return update

    async def _lookup(self, order_id: str, context: TurnContext) -> Order | None:
        order = await self._order_tool.ainvoke({"order_id": order_id}, _tool_config(context))
        return cast(Order | None, order)


# --- edges and helpers -----------------------------------------------------------------------


def _outage_safe(step: _Step) -> _Step:
    """A provider outage in a step ends the turn with the contact channels and the previous
    state (NFR-11), inside the graph, so the turn still joins the transcript."""

    async def safe(state: TurnState, *, runtime: Runtime[TurnContext]) -> Update:
        try:
            return await step(state, runtime=runtime)
        except ProviderUnavailableError as exc:
            log.error("provider unavailable: %s", exc)
            return {"response": outage_contact_reply()}

    return safe


def _answered_or(next_node: str) -> Callable[[TurnState], str]:
    def route(state: TurnState) -> str:
        return "finish" if state["response"] is not None else next_node

    return route


def _route_step(state: TurnState) -> str:
    match _decision(state).step:
        case NextStep.ORDER_GUIDANCE:
            return "order_guidance"
        case NextStep.ANSWER_FROM_KNOWLEDGE:
            return "knowledge"
        case NextStep.CONFIRM_TICKET:
            return "confirm_ticket"
        case NextStep.CREATE_TICKET:
            return "create_ticket"
        case _:
            return "template_reply"


def _routing(state: TurnState) -> RoutingResult:
    routing = state["routing"]
    if routing is None:  # pragma: no cover - the gate ends such turns
        raise RuntimeError("a step without a routing")
    return routing


def _decision(state: TurnState) -> Decision:
    decision = state["decision"]
    if decision is None:  # pragma: no cover - steps run after decide
        raise RuntimeError("a step without a decision")
    return decision


def _conversation(state: TurnState, runtime: Runtime[TurnContext]) -> Conversation:
    """What the models see: the latest turns of the transcript, this message last."""
    return Conversation(
        messages=state["messages"][-MAX_CONTEXT_MESSAGES:],
        customer_id=runtime.context.customer_id,
    )


def _tool_config(context: TurnContext) -> RunnableConfig:
    return tool_config(
        customer_id=context.customer_id, tenant_id=context.tenant_id, session_id=context.session_id
    )


def _done(state: TurnState, response: AgentResponse, session: SessionState | None = None) -> Update:
    """A step's reply carries the step's intent; the session moves to the step's new state."""
    decision = _decision(state)
    final = response.model_copy(update={"intent": decision.intent})
    log.info("behavior=%s sources=%s", final.behavior, ",".join(final.sources) or "-")
    return {"response": final, "session": session or decision.state}


def _payment_data(screen: GuardrailDecision) -> bool:
    """Payment data was masked: the customer is told so before and with the reply."""
    return not PAYMENT_DATA_REASONS.isdisjoint(screen.reasons)


def _appended(response: AgentResponse, text: str) -> AgentResponse:
    return response.model_copy(update={"answer": f"{response.answer}\n\n{text}"})


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
