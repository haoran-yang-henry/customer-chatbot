"""Explicit conversation state and the state machine that decides the next step (ADR-0005).

``decide`` is a pure function: (state, routing result, authenticated?) → next step + new state.
Every transition is covered by unit tests; no LLM is involved.
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


class SessionState(BaseModel):
    model_config = ConfigDict(frozen=True)

    active_intent: Intent | None = None
    workflow: WorkflowState = WorkflowState.IDLE
    slots: dict[str, str] = Field(default_factory=dict)


class NextStep(StrEnum):
    ANSWER_FROM_KNOWLEDGE = "answer_from_knowledge"
    ORDER_GUIDANCE = "order_guidance"  # Day 5: order lookup + eligibility; Day 4: policy answer
    ASK_ORDER_ID = "ask_order_id"
    ASK_LOGIN = "ask_login"
    HANDOFF = "handoff"  # authenticated: ticket for a human colleague
    HANDOFF_CONTACT = "handoff_contact"  # anonymous: contact channels, no untraceable ticket
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
        # The agent cannot grant money (ADR-0006). Complaints and explicit handoff wishes still
        # reach a human, who can decide on goodwill (S-007, T07).
        return Decision(step=NextStep.REFUSE_MONEY_DEMAND, intent=intent, state=state)

    idle = state.model_copy(update={"active_intent": intent, "workflow": WorkflowState.IDLE})

    match intent.group:
        case IntentGroup.CONTROL:
            step = NextStep.GREET if intent is Intent.GREETING else NextStep.REFUSE_OUT_OF_SCOPE
            return Decision(step=step, intent=intent, state=idle)
        case IntentGroup.SUPPORT:
            step = NextStep.HANDOFF if authenticated else NextStep.HANDOFF_CONTACT
            return Decision(step=step, intent=intent, state=idle)
        case IntentGroup.KNOWLEDGE:
            return Decision(step=NextStep.ANSWER_FROM_KNOWLEDGE, intent=intent, state=idle)

    # Transactional: about one concrete order of the customer, or a general how-to question?
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
