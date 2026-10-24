"""Explicit conversation state and the state machine that decides the next step (ADR-0005).

``decide`` is a pure function: (state, routing result, authenticated?) → next step + new state.
Every transition is covered by unit tests; no LLM is involved.

Tickets (ADR-0014): only a request to change stored data becomes a ticket, and only after the
customer confirmed it once. Questions get answers, a wish to talk to a person without such a
request gets the contact details, and one conversation opens at most one ticket.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from customer_agent.domain.intents import Intent, IntentGroup
from customer_agent.domain.routing import RoutingResult, RoutingSource

# Below this router confidence the agent asks instead of guessing (taxonomy §4).
CONFIDENCE_THRESHOLD = 0.6


class WorkflowState(StrEnum):
    IDLE = "idle"
    AWAITING_ORDER_ID = "awaiting_order_id"
    AWAITING_TICKET_CONFIRMATION = "awaiting_ticket_confirmation"


class SessionState(BaseModel):
    model_config = ConfigDict(frozen=True)

    active_intent: Intent | None = None
    workflow: WorkflowState = WorkflowState.IDLE
    # order_id: the order being discussed; request / question: the change awaiting
    # confirmation and the question asked about it; ticket_id: the ticket already opened.
    slots: dict[str, str] = Field(default_factory=dict)


class NextStep(StrEnum):
    ANSWER_FROM_KNOWLEDGE = "answer_from_knowledge"
    ORDER_GUIDANCE = "order_guidance"  # Day 5: order lookup + eligibility; Day 4: policy answer
    ASK_ORDER_ID = "ask_order_id"
    ASK_LOGIN = "ask_login"
    CONFIRM_TICKET = "confirm_ticket"  # a change request: ask once before opening a ticket
    CREATE_TICKET = "create_ticket"  # confirmed: ticket for the service team
    TICKET_DECLINED = "ticket_declined"
    TICKET_EXISTS = "ticket_exists"  # this conversation already opened one: no second ticket
    CONTACT = "contact"  # no change request, or anonymous: contact details, no ticket
    GREET = "greet"
    REFUSE_OUT_OF_SCOPE = "refuse_out_of_scope"
    REFUSE_THIRD_PARTY = "refuse_third_party"
    REFUSE_MONEY_DEMAND = "refuse_money_demand"
    CLARIFY = "clarify"


class Decision(BaseModel):
    model_config = ConfigDict(frozen=True)

    step: NextStep
    intent: Intent | None
    state: SessionState


def decide(state: SessionState, routing: RoutingResult, *, authenticated: bool) -> Decision:
    if state.workflow is WorkflowState.AWAITING_TICKET_CONFIRMATION:
        reply = _answer_to_question(state, routing)
        if reply is not None:
            return reply
        # Anything else is a new topic: the question lapses, nothing is forwarded.
        state = _without_request(state)
    if routing.source is RoutingSource.FALLBACK:
        return Decision(step=NextStep.ANSWER_FROM_KNOWLEDGE, intent=None, state=state)
    if routing.third_party:
        # Other people's orders or data: refused before any lookup (privacy, T06).
        return Decision(step=NextStep.REFUSE_THIRD_PARTY, intent=routing.intent, state=state)
    intent = routing.intent
    if intent is None or routing.confidence < CONFIDENCE_THRESHOLD:
        # Understand first, refuse second: "Mein Geld!" is a question to clarify (A-005).
        return Decision(step=NextStep.CLARIFY, intent=intent, state=state)
    if routing.money_demand and intent.group is not IntentGroup.SUPPORT:
        # The agent cannot grant money (ADR-0006), whatever is claimed (S-007, T07).
        return Decision(step=NextStep.REFUSE_MONEY_DEMAND, intent=intent, state=state)

    idle = state.model_copy(update={"active_intent": intent, "workflow": WorkflowState.IDLE})

    if intent.group is IntentGroup.CONTROL:
        step = NextStep.GREET if intent is Intent.GREETING else NextStep.REFUSE_OUT_OF_SCOPE
        return Decision(step=step, intent=intent, state=idle)
    if routing.change_request and intent.group is not IntentGroup.TRANSACTIONAL:
        # Order changes take the transactional path: lookup and eligibility come first.
        return _service_team_change(idle, routing, intent, authenticated=authenticated)
    match intent.group:
        case IntentGroup.SUPPORT:
            # Nothing to change: a complaint gets the relevant information, a wish to talk to
            # a person the contact details. No ticket either way.
            if intent is Intent.COMPLAINT:
                return Decision(step=NextStep.ANSWER_FROM_KNOWLEDGE, intent=intent, state=idle)
            return Decision(step=NextStep.CONTACT, intent=intent, state=idle)
        case IntentGroup.KNOWLEDGE:
            return Decision(step=NextStep.ANSWER_FROM_KNOWLEDGE, intent=intent, state=idle)

    return _transactional(state, idle, routing, intent, authenticated=authenticated)


def _transactional(
    state: SessionState,
    idle: SessionState,
    routing: RoutingResult,
    intent: Intent,
    *,
    authenticated: bool,
) -> Decision:
    """About one concrete order of the customer, or a general how-to question?"""
    remembered = state.slots.get("order_id") if routing.about_specific_order else None
    order_id = routing.order_id or remembered
    if order_id is None and not routing.about_specific_order:
        return Decision(step=NextStep.ANSWER_FROM_KNOWLEDGE, intent=intent, state=idle)
    if not authenticated:
        return Decision(step=NextStep.ASK_LOGIN, intent=intent, state=idle)
    if order_id is None:
        waiting = idle.model_copy(update={"workflow": WorkflowState.AWAITING_ORDER_ID})
        return Decision(step=NextStep.ASK_ORDER_ID, intent=intent, state=waiting)
    with_order = idle.model_copy(update={"slots": {**state.slots, "order_id": order_id}})
    return Decision(step=NextStep.ORDER_GUIDANCE, intent=intent, state=with_order)


def _service_team_change(
    idle: SessionState, routing: RoutingResult, intent: Intent, *, authenticated: bool
) -> Decision:
    """Data only the service team can change (account, address, …): ask once, then ticket."""
    if not authenticated:
        return Decision(step=NextStep.CONTACT, intent=intent, state=idle)
    return _offer_ticket(idle, routing, intent)


def _offer_ticket(state: SessionState, routing: RoutingResult, intent: Intent) -> Decision:
    if "ticket_id" in state.slots:
        return Decision(step=NextStep.TICKET_EXISTS, intent=intent, state=state)
    slots = {**state.slots, "request": routing.standalone_query}
    if routing.order_id:
        slots["order_id"] = routing.order_id  # verified by a lookup before the question
    pending = state.model_copy(
        update={"workflow": WorkflowState.AWAITING_TICKET_CONFIRMATION, "slots": slots}
    )
    return Decision(step=NextStep.CONFIRM_TICKET, intent=intent, state=pending)


def _answer_to_question(state: SessionState, routing: RoutingResult) -> Decision | None:
    """The reply while "Soll ich … weitergeben?" is open: yes, no, or a follow-up question;
    ``None`` for a new topic."""
    reply = _confirmation(state, routing)
    if reply is not None:
        return reply
    if _follow_up(state, routing):
        # A question about the same matter ("Wie sende ich das zurück?"): answer it from the
        # documents; the question stays open, so a later "Ja" still forwards the request.
        return Decision(
            step=NextStep.ANSWER_FROM_KNOWLEDGE, intent=state.active_intent, state=state
        )
    return None


def _confirmation(state: SessionState, routing: RoutingResult) -> Decision | None:
    """The customer's answer to "Soll ich … an unser Service-Team weitergeben?"."""
    intent = state.active_intent or routing.intent
    same_matter = routing.intent is not None and routing.intent is state.active_intent
    agrees = routing.confirmation is True or (
        routing.confirmation is None
        and (routing.intent is Intent.HUMAN_AGENT or (routing.change_request and same_matter))
    )
    if agrees:
        if "ticket_id" in state.slots:
            return Decision(step=NextStep.TICKET_EXISTS, intent=intent, state=state)
        slots = {k: v for k, v in state.slots.items() if k != "question"}
        confirmed = state.model_copy(update={"workflow": WorkflowState.IDLE, "slots": slots})
        return Decision(step=NextStep.CREATE_TICKET, intent=intent, state=confirmed)
    if routing.confirmation is False:
        return Decision(step=NextStep.TICKET_DECLINED, intent=intent, state=_without_request(state))
    return None


def _follow_up(state: SessionState, routing: RoutingResult) -> bool:
    other_order = routing.order_id is not None and routing.order_id != state.slots.get("order_id")
    return (
        routing.source is RoutingSource.LLM
        and routing.intent is not None
        and routing.intent is state.active_intent
        and not other_order
    )


def _without_request(state: SessionState) -> SessionState:
    slots = {k: v for k, v in state.slots.items() if k not in ("request", "question")}
    return state.model_copy(update={"workflow": WorkflowState.IDLE, "slots": slots})
