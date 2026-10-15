"""Every transition of the state machine (ADR-0005). No LLM, no I/O."""

import pytest

from customer_agent.domain.intents import Intent
from customer_agent.domain.routing import RoutingResult, RoutingSource, extract_order_id
from customer_agent.domain.session import (
    CONFIDENCE_THRESHOLD,
    NextStep,
    SessionState,
    WorkflowState,
    decide,
)


def routed(
    intent: Intent | None,
    *,
    confidence: float = 0.9,
    order_id: str | None = None,
    specific: bool = False,
    change: bool = False,
    confirmation: bool | None = None,
    source: RoutingSource = RoutingSource.LLM,
) -> RoutingResult:
    return RoutingResult(
        intent=intent,
        confidence=confidence,
        order_id=order_id,
        about_specific_order=specific or order_id is not None,
        change_request=change,
        confirmation=confirmation,
        standalone_query="q",
        source=source,
    )


IDLE = SessionState()
WAITING = SessionState(active_intent=Intent.ORDER_STATUS, workflow=WorkflowState.AWAITING_ORDER_ID)


@pytest.mark.parametrize(
    ("intent", "step"),
    [
        (Intent.RETURN_POLICY, NextStep.ANSWER_FROM_KNOWLEDGE),
        (Intent.PRODUCT_INFO, NextStep.ANSWER_FROM_KNOWLEDGE),
        (Intent.GREETING, NextStep.GREET),
        (Intent.OUT_OF_SCOPE, NextStep.REFUSE_OUT_OF_SCOPE),
        # ADR-0014: no change request, no ticket — information or the contact details.
        (Intent.COMPLAINT, NextStep.ANSWER_FROM_KNOWLEDGE),
        (Intent.HUMAN_AGENT, NextStep.CONTACT),
    ],
)
def test_non_transactional_intents(intent: Intent, step: NextStep) -> None:
    decision = decide(IDLE, routed(intent), authenticated=True)

    assert decision.step is step
    assert decision.state.active_intent is intent
    assert decision.state.workflow is WorkflowState.IDLE


@pytest.mark.parametrize("confidence", [0.0, CONFIDENCE_THRESHOLD - 0.01])
def test_low_confidence_asks_instead_of_guessing(confidence: float) -> None:
    decision = decide(IDLE, routed(Intent.ORDER_STATUS, confidence=confidence), authenticated=True)

    assert decision.step is NextStep.CLARIFY
    assert decision.state == IDLE  # nothing is committed on a guess


def test_unknown_intent_asks() -> None:
    assert decide(IDLE, routed(None), authenticated=True).step is NextStep.CLARIFY


def test_router_fallback_answers_from_knowledge() -> None:
    decision = decide(IDLE, routed(None, source=RoutingSource.FALLBACK), authenticated=True)

    assert decision.step is NextStep.ANSWER_FROM_KNOWLEDGE


def test_general_transactional_question_is_answered_from_knowledge() -> None:
    # "Wie kann ich eine Bestellung stornieren?" — allowed even when anonymous (FR-20).
    decision = decide(IDLE, routed(Intent.ORDER_CANCEL), authenticated=False)

    assert decision.step is NextStep.ANSWER_FROM_KNOWLEDGE


@pytest.mark.parametrize("order_id", [None, "NE-100250"])
def test_anonymous_customer_must_log_in_for_own_orders(order_id: str | None) -> None:
    decision = decide(
        IDLE, routed(Intent.ORDER_STATUS, order_id=order_id, specific=True), authenticated=False
    )

    assert decision.step is NextStep.ASK_LOGIN
    assert "order_id" not in decision.state.slots


def test_missing_order_number_is_requested_and_remembered_as_waiting() -> None:
    decision = decide(IDLE, routed(Intent.ORDER_CANCEL, specific=True), authenticated=True)

    assert decision.step is NextStep.ASK_ORDER_ID
    assert decision.state.workflow is WorkflowState.AWAITING_ORDER_ID
    assert decision.state.active_intent is Intent.ORDER_CANCEL


def test_order_number_completes_the_waiting_workflow() -> None:
    decision = decide(
        WAITING, routed(Intent.ORDER_STATUS, order_id="NE-100250"), authenticated=True
    )

    assert decision.step is NextStep.ORDER_GUIDANCE
    assert decision.state.workflow is WorkflowState.IDLE
    assert decision.state.slots == {"order_id": "NE-100250"}


def test_remembered_order_is_reused_for_follow_up_about_the_same_order() -> None:
    known = SessionState(active_intent=Intent.ORDER_STATUS, slots={"order_id": "NE-100250"})

    decision = decide(known, routed(Intent.ORDER_CANCEL, specific=True), authenticated=True)

    assert decision.step is NextStep.ORDER_GUIDANCE
    assert decision.state.slots["order_id"] == "NE-100250"


def test_third_party_request_is_refused_regardless_of_intent_or_login() -> None:
    routing = routed(Intent.ORDER_CANCEL, order_id="NE-100260").model_copy(
        update={"third_party": True}
    )

    decision = decide(IDLE, routing, authenticated=True)

    assert decision.step is NextStep.REFUSE_THIRD_PARTY
    assert decision.state == IDLE


@pytest.mark.parametrize("intent", [Intent.REFUND_STATUS, Intent.PRODUCT_INFO, Intent.ORDER_CANCEL])
def test_money_demand_is_refused_before_any_lookup(intent: Intent) -> None:
    routing = routed(intent, order_id="NE-100231").model_copy(update={"money_demand": True})

    decision = decide(IDLE, routing, authenticated=True)

    assert decision.step is NextStep.REFUSE_MONEY_DEMAND
    assert decision.state == IDLE


@pytest.mark.parametrize(("intent", "confidence"), [(None, 0.9), (Intent.REFUND_STATUS, 0.3)])
def test_unclear_money_message_is_clarified_not_refused(
    intent: Intent | None, confidence: float
) -> None:
    routing = routed(intent, confidence=confidence).model_copy(update={"money_demand": True})

    assert decide(IDLE, routing, authenticated=True).step is NextStep.CLARIFY


@pytest.mark.parametrize(
    ("intent", "step"),
    [(Intent.COMPLAINT, NextStep.ANSWER_FROM_KNOWLEDGE), (Intent.HUMAN_AGENT, NextStep.CONTACT)],
)
def test_money_demand_in_a_complaint_opens_no_ticket(intent: Intent, step: NextStep) -> None:
    routing = routed(intent).model_copy(update={"money_demand": True})

    assert decide(IDLE, routing, authenticated=True).step is step


def test_topic_switch_while_waiting_abandons_the_workflow_explicitly() -> None:
    decision = decide(WAITING, routed(Intent.SHIPPING_INFO), authenticated=True)

    assert decision.step is NextStep.ANSWER_FROM_KNOWLEDGE
    assert decision.state.workflow is WorkflowState.IDLE  # FR-04: not silently kept


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Wo ist NE-100250?", "NE-100250"),
        ("bestellung ne 100245 bitte", "NE-100245"),
        ("NE100260", "NE-100260"),
        ("Nummer 100250", None),
        ("NE-12345", None),
    ],
)
def test_order_id_extraction(text: str, expected: str | None) -> None:
    assert extract_order_id(text) == expected


# --- tickets (ADR-0014) ------------------------------------------------------------------------

PENDING = SessionState(
    active_intent=Intent.ORDER_CHANGE,
    workflow=WorkflowState.AWAITING_TICKET_CONFIRMATION,
    slots={"order_id": "NE-100260", "request": "Lieferadresse für NE-100260 ändern"},
)


@pytest.mark.parametrize("intent", [Intent.HUMAN_AGENT, Intent.SERVICE_INFO, Intent.COMPLAINT])
def test_a_change_request_is_confirmed_once_before_a_ticket(intent: Intent) -> None:
    decision = decide(IDLE, routed(intent, change=True, order_id="NE-100260"), authenticated=True)

    assert decision.step is NextStep.CONFIRM_TICKET
    assert decision.state.workflow is WorkflowState.AWAITING_TICKET_CONFIRMATION
    assert decision.state.slots == {"order_id": "NE-100260", "request": "q"}


@pytest.mark.parametrize("change", [False, True])
def test_anonymous_customers_get_contact_details_never_a_ticket(change: bool) -> None:
    for intent in (Intent.HUMAN_AGENT, Intent.SERVICE_INFO):
        decision = decide(IDLE, routed(intent, change=change), authenticated=False)
        if intent is Intent.SERVICE_INFO and not change:
            assert decision.step is NextStep.ANSWER_FROM_KNOWLEDGE
        else:
            assert decision.step is NextStep.CONTACT


def test_an_order_change_takes_the_order_path_first() -> None:
    specific = routed(Intent.ORDER_CHANGE, order_id="NE-100260", change=True)
    general = routed(Intent.ORDER_CHANGE)  # "Wie kann ich die Lieferadresse ändern?"

    assert decide(IDLE, specific, authenticated=True).step is NextStep.ORDER_GUIDANCE
    assert decide(IDLE, general, authenticated=True).step is NextStep.ANSWER_FROM_KNOWLEDGE
    unknown_order = routed(Intent.ORDER_CHANGE, specific=True, change=True)
    assert decide(IDLE, unknown_order, authenticated=True).step is NextStep.ASK_ORDER_ID


@pytest.mark.parametrize(
    "routing",
    [
        routed(None, confirmation=True, source=RoutingSource.RULE),
        routed(Intent.HUMAN_AGENT),  # "Ja, leiten Sie das bitte an einen Mitarbeiter weiter"
    ],
)
def test_agreement_opens_the_ticket_for_the_pending_request(routing: RoutingResult) -> None:
    decision = decide(PENDING, routing, authenticated=True)

    assert decision.step is NextStep.CREATE_TICKET
    assert decision.intent is Intent.ORDER_CHANGE  # what was asked about, not "Ja"
    assert decision.state.slots["request"] == "Lieferadresse für NE-100260 ändern"
    assert decision.state.workflow is WorkflowState.IDLE


def test_declining_forwards_nothing() -> None:
    decision = decide(PENDING, routed(None, confirmation=False), authenticated=True)

    assert decision.step is NextStep.TICKET_DECLINED
    assert "request" not in decision.state.slots
    assert decision.state.workflow is WorkflowState.IDLE


def test_a_new_topic_lets_the_question_lapse() -> None:
    decision = decide(PENDING, routed(Intent.SHIPPING_INFO), authenticated=True)

    assert decision.step is NextStep.ANSWER_FROM_KNOWLEDGE
    assert decision.state.workflow is WorkflowState.IDLE and "request" not in decision.state.slots


def test_one_conversation_opens_at_most_one_ticket() -> None:
    done = SessionState(active_intent=Intent.ORDER_CHANGE, slots={"ticket_id": "T-000007"})
    pending_after_ticket = PENDING.model_copy(
        update={"slots": {**PENDING.slots, "ticket_id": "T-000007"}}
    )

    another = decide(done, routed(Intent.HUMAN_AGENT, change=True), authenticated=True)
    agreed = decide(pending_after_ticket, routed(None, confirmation=True), authenticated=True)

    assert another.step is NextStep.TICKET_EXISTS
    assert agreed.step is NextStep.TICKET_EXISTS


def test_a_follow_up_on_the_same_matter_keeps_the_question_open() -> None:
    decision = decide(PENDING, routed(Intent.ORDER_CHANGE), authenticated=True)

    assert decision.step is NextStep.ANSWER_FROM_KNOWLEDGE
    assert decision.state == PENDING  # a later "Ja" still forwards the request


def test_a_follow_up_about_another_order_lets_the_question_lapse() -> None:
    decision = decide(
        PENDING, routed(Intent.ORDER_CHANGE, order_id="NE-100252"), authenticated=True
    )

    assert decision.step is NextStep.ORDER_GUIDANCE
    assert decision.state.workflow is WorkflowState.IDLE and "request" not in decision.state.slots


def test_asking_for_the_change_itself_while_asked_is_agreement() -> None:
    decision = decide(PENDING, routed(Intent.ORDER_CHANGE, change=True), authenticated=True)

    assert decision.step is NextStep.CREATE_TICKET
