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
    source: RoutingSource = RoutingSource.LLM,
) -> RoutingResult:
    return RoutingResult(
        intent=intent,
        confidence=confidence,
        order_id=order_id,
        about_specific_order=specific or order_id is not None,
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
        (Intent.COMPLAINT, NextStep.HANDOFF),
        (Intent.HUMAN_AGENT, NextStep.HANDOFF),
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


@pytest.mark.parametrize("intent", [Intent.HUMAN_AGENT, Intent.COMPLAINT])
def test_anonymous_handoff_gives_contact_channels_instead_of_a_ticket(intent: Intent) -> None:
    assert decide(IDLE, routed(intent), authenticated=False).step is NextStep.HANDOFF_CONTACT
