"""Stage 2 of a turn (rebuild plan §4.2): what to do with an understood message, as a LangGraph.

    START → gate ─ outage or veto ───────────────────────────────────────────→ END
              └→ decide ─ ORDER_GUIDANCE ────→ order_guidance ─ already given → knowledge
                        ├ ANSWER_FROM_KNOWLEDGE → knowledge ──────────────────────→ END
                        ├ CONFIRM_TICKET ─────→ confirm_ticket ───────────────────→ END
                        ├ CREATE_TICKET ──────→ create_ticket ────────────────────→ END
                        └ every other step ───→ template_reply ───────────────────→ END

``decide()`` stays the pure state machine of ``domain/session.py``; the graph only routes on its
``Decision``. The state holds the masked conversation only (the input check ran in stage 1).
Guardrails judge; the nodes decide what the customer sees (ADR-0011). Tickets only for
confirmed change requests, at most one per conversation (ADR-0014). The answer's tokens leave
through the ``custom`` stream; the response is in the final state.
"""

from dataclasses import dataclass
from typing import Any, NotRequired, TypedDict, cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from customer_agent.application.answering import AnswerService, FinalAnswer
from customer_agent.application.fallbacks import (
    confirm_ticket_reply,
    contact_reply,
    fallback,
    outage_contact_reply,
    step_reply,
    ticket_created_reply,
    ticket_exists_reply,
    ticket_question,
)
from customer_agent.application.order_guidance import order_guidance, order_not_found
from customer_agent.application.ports import Clock
from customer_agent.application.tools import (
    CreateTicket,
    GetOrderStatus,
    TicketOutcome,
    tool_config,
)
from customer_agent.domain.conversation import AgentResponse, Conversation, Role
from customer_agent.domain.guardrails import GuardrailDecision
from customer_agent.domain.intents import Intent
from customer_agent.domain.orders import Order
from customer_agent.domain.routing import RoutingResult
from customer_agent.domain.session import Decision, NextStep, SessionState, WorkflowState, decide
from customer_agent.observability import get_logger, record_guardrail

log = get_logger("chat")

# Facts the customer may simply ask for again ("Und wo ist sie jetzt?"); everything else is
# guidance, which is not repeated word for word.
_FACT_LOOKUPS = {Intent.ORDER_STATUS, Intent.REFUND_STATUS}


class TurnState(TypedDict):
    conversation: Conversation  # masked; the last message is this turn's
    session: SessionState
    routing: RoutingResult | None  # None: the router's provider failed (NFR-11)
    verdict: GuardrailDecision | None
    decision: NotRequired[Decision]
    response: NotRequired[AgentResponse]


@dataclass(frozen=True)
class TurnContext:
    session_id: str
    tenant_id: str


Update = dict[str, Any]


class TurnGraph:
    def __init__(
        self,
        answers: AnswerService,
        order_tool: GetOrderStatus,
        ticket_tool: CreateTicket,
        clock: Clock,
    ) -> None:
        self._answers = answers
        self._order_tool = order_tool
        self._ticket_tool = ticket_tool
        self._clock = clock
        self.graph = self._build()

    def _build(self) -> CompiledStateGraph[TurnState, TurnContext, TurnState, TurnState]:
        graph = StateGraph(TurnState, context_schema=TurnContext)
        graph.add_node("gate", self._gate)
        graph.add_node("decide", self._decide)
        graph.add_node("order_guidance", self._order_guidance)
        graph.add_node("knowledge", self._knowledge)
        graph.add_node("confirm_ticket", self._confirm_ticket)
        graph.add_node("create_ticket", self._create_ticket)
        graph.add_node("template_reply", self._template_reply)
        graph.add_edge(START, "gate")
        graph.add_conditional_edges("gate", _answered_or("decide"), ["decide", END])
        graph.add_conditional_edges(
            "decide",
            _route_step,
            ["order_guidance", "knowledge", "confirm_ticket", "create_ticket", "template_reply"],
        )
        graph.add_conditional_edges("order_guidance", _answered_or("knowledge"), ["knowledge", END])
        for node in ("knowledge", "confirm_ticket", "create_ticket", "template_reply"):
            graph.add_edge(node, END)
        return graph.compile(name="turn")

    # --- nodes -------------------------------------------------------------------------------

    def _gate(self, state: TurnState) -> Update:
        """Nothing understood, or the classifier vetoes: a reply without any step."""
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

    def _decide(self, state: TurnState) -> Update:
        authenticated = state["conversation"].customer_id is not None
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
        decision, conversation = state["decision"], state["conversation"]
        session = decision.state
        order_id = session.slots["order_id"]
        intent = decision.intent or Intent.ORDER_STATUS
        order = await self._lookup(conversation, order_id, runtime.context)
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
        decision = state["decision"]
        pending = decision.state.slots.get("question") if _awaiting(decision.state) else None
        response: AgentResponse | None = None
        async for event in self._answers.answer_stream(
            state["conversation"],
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
        decision = state["decision"]
        session = decision.state
        order_id = session.slots.get("order_id")
        if order_id is not None:
            order = await self._lookup(state["conversation"], order_id, runtime.context)
            if order is None:
                slots = {k: v for k, v in session.slots.items() if k not in ("order_id", "request")}
                idle = session.model_copy(update={"workflow": WorkflowState.IDLE, "slots": slots})
                intent = decision.intent or Intent.HUMAN_AGENT
                return _done(state, order_not_found(order_id, intent), idle)
        question = ticket_question(decision.intent, order_id)
        asked = session.model_copy(update={"slots": {**session.slots, "question": question}})
        return _done(state, confirm_ticket_reply(decision.intent, order_id), asked)

    async def _create_ticket(self, state: TurnState, runtime: Runtime[TurnContext]) -> Update:
        decision, conversation = state["decision"], state["conversation"]
        session = decision.state
        arguments = {
            "intent": decision.intent,
            "order_id": session.slots.get("order_id"),
            # The confirmed request, not the confirmation itself ("Ja, bitte").
            "request": session.slots.get("request") or conversation.messages[-1].content,
        }
        outcome = cast(
            TicketOutcome,
            await self._ticket_tool.ainvoke(arguments, config=_tools(conversation, runtime)),
        )
        slots = {k: v for k, v in session.slots.items() if k != "request"}
        done = session.model_copy(update={"slots": {**slots, "ticket_id": outcome.ticket_id}})
        # At the daily limit the tool returns the newest open ticket instead of a new one.
        reply = ticket_created_reply if outcome.created else ticket_exists_reply
        return _done(state, reply(outcome.ticket_id, intent=decision.intent), done)

    def _template_reply(self, state: TurnState) -> Update:
        decision = state["decision"]
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

    async def _lookup(
        self, conversation: Conversation, order_id: str, context: TurnContext
    ) -> Order | None:
        config = tool_config(
            customer_id=conversation.customer_id,
            tenant_id=context.tenant_id,
            session_id=context.session_id,
        )
        return cast(Order | None, await self._order_tool.ainvoke({"order_id": order_id}, config))


# --- edges and helpers -----------------------------------------------------------------------


def _answered_or(next_node: str) -> Any:
    def route(state: TurnState) -> str:
        return END if "response" in state else next_node

    return route


def _route_step(state: TurnState) -> str:
    match state["decision"].step:
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


def _tools(conversation: Conversation, runtime: Runtime[TurnContext]) -> RunnableConfig:
    return tool_config(
        customer_id=conversation.customer_id,
        tenant_id=runtime.context.tenant_id,
        session_id=runtime.context.session_id,
    )


def _done(state: TurnState, response: AgentResponse, session: SessionState | None = None) -> Update:
    """A step's reply carries the step's intent; the session moves to the step's new state."""
    decision = state["decision"]
    final = response.model_copy(update={"intent": decision.intent})
    log.info("behavior=%s sources=%s", final.behavior, ",".join(final.sources) or "-")
    return {"response": final, "session": session or decision.state}


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
