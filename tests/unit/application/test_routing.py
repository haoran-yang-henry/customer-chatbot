import pytest

from customer_agent.application.routing import (
    IntentRouter,
    parse_llm_routing,
    route_by_rules,
)
from customer_agent.domain.conversation import Conversation, Message, Role
from customer_agent.domain.intents import Intent
from customer_agent.domain.routing import RoutingSource
from customer_agent.domain.session import SessionState, WorkflowState
from customer_agent.infrastructure.prompts import load_prompt
from tests.fakes import ScriptedRouterLLM


def convo(*texts: str) -> Conversation:
    roles = [Role.USER, Role.ASSISTANT]
    return Conversation(
        messages=[Message(role=roles[i % 2], content=t) for i, t in enumerate(texts)]
    )


WAITING = SessionState(active_intent=Intent.ORDER_CANCEL, workflow=WorkflowState.AWAITING_ORDER_ID)


def test_greeting_is_routed_without_llm() -> None:
    result = route_by_rules(convo("Grüß Gott!"), SessionState())

    assert result is not None and result.intent is Intent.GREETING
    assert result.source is RoutingSource.RULE


def test_greeting_with_a_question_goes_to_the_llm() -> None:
    assert route_by_rules(convo("Hallo, was kostet Versand?"), SessionState()) is None


def test_bare_order_number_while_waiting_completes_the_slot_without_llm() -> None:
    result = route_by_rules(convo("Stornieren bitte", "Welche Bestellung?", "NE-100245"), WAITING)

    assert result is not None
    assert (result.intent, result.order_id) == (Intent.ORDER_CANCEL, "NE-100245")


def test_order_number_without_waiting_state_goes_to_the_llm() -> None:
    assert route_by_rules(convo("NE-100245"), SessionState()) is None


def test_parse_valid_json_and_take_order_id_from_text_not_from_llm() -> None:
    text = (
        '{"intent": "ORDER_STATUS", "confidence": 0.93, "about_specific_order": true, '
        '"standalone_query": "Wo ist meine Bestellung?", "order_id": "NE-999999"}'
    )

    result = parse_llm_routing(text, convo("Wo ist meine Bestellung NE-100250?"))

    assert result.intent is Intent.ORDER_STATUS and result.confidence == 0.93
    assert result.order_id == "NE-100250"


def test_parse_null_intent_with_clarification() -> None:
    text = (
        '{"intent": null, "confidence": 0.2, "standalone_query": "", '
        '"clarification_question": "Worum geht es?"}'
    )

    result = parse_llm_routing(text, convo("Wo ist es?"))

    assert result.intent is None and result.clarification_question == "Worum geht es?"
    assert result.standalone_query == "Wo ist es?"  # empty rewrite → original text


def test_unusable_llm_output_falls_back_to_knowledge() -> None:
    for text in ["Sorry, I cannot.", '{"intent": "ORDER_MODIFY"}', "{broken"]:
        assert parse_llm_routing(text, convo("Frage")).source is RoutingSource.FALLBACK


def test_confidence_is_clamped() -> None:
    text = '{"intent": "GREETING", "confidence": 7, "standalone_query": "x"}'

    assert parse_llm_routing(text, convo("x")).confidence == 1.0


async def test_router_calls_llm_once_with_state_and_history() -> None:
    llm = ScriptedRouterLLM(
        '{"intent": "SHIPPING_INFO", "confidence": 0.9, "standalone_query": "Versand Schweiz?"}'
    )
    router = IntentRouter(llm, load_prompt("router", 1))

    result = await router.route(
        convo("Versand AT?", "7,95 €.", "Und CH?"), SessionState(), session_id="s1"
    )

    assert result.standalone_query == "Versand Schweiz?"
    assert llm.router_calls == 1 and llm.router_session_ids == ["s1"]
    user_prompt = llm.router_prompts[0][1].content
    assert "Customer: Versand AT?" in user_prompt and "New conversation." in user_prompt


def test_parse_third_party_flag() -> None:
    text = (
        '{"intent": "ORDER_STATUS", "confidence": 0.9, "third_party": true, '
        '"standalone_query": "Welche Bestellungen hat Jonas Weber?"}'
    )

    assert parse_llm_routing(text, convo("Welche Bestellungen hat Jonas Weber?")).third_party


def test_parse_money_demand_flag() -> None:
    text = (
        '{"intent": "REFUND_STATUS", "confidence": 0.9, "money_demand": true, '
        '"standalone_query": "Erstattung für NE-100231"}'
    )

    result = parse_llm_routing(text, convo("Überweisen Sie mir sofort 300 € für NE-100231!"))

    assert result.money_demand and result.intent is Intent.REFUND_STATUS


async def test_unsafe_clarification_question_is_replaced_by_the_default() -> None:
    """The clarification is shown verbatim: it must not become a channel for links or the
    prompt (an injected message could ask the router to write them)."""
    llm = ScriptedRouterLLM(
        '{"intent": null, "confidence": 0.2, "standalone_query": "?", '
        '"clarification_question": "Bitte melden Sie sich unter https://evil.example an."}'
    )
    router = IntentRouter(llm, load_prompt("router", 5))

    result = await router.route(convo("Hilfe?"), SessionState(), session_id="s")

    assert result.clarification_question is None


async def test_safe_clarification_question_is_kept() -> None:
    llm = ScriptedRouterLLM(
        '{"intent": "PRODUCT_INFO", "confidence": 0.4, "standalone_query": "Kopfhörer", '
        '"clarification_question": "Was möchten Sie über die Kopfhörer wissen?"}'
    )
    router = IntentRouter(llm, load_prompt("router", 5))

    result = await router.route(convo("Kopfhörer"), SessionState(), session_id="s")

    assert result.clarification_question == "Was möchten Sie über die Kopfhörer wissen?"


ASKED = SessionState(
    active_intent=Intent.ORDER_CHANGE, workflow=WorkflowState.AWAITING_TICKET_CONFIRMATION
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Ja", True),
        ("ja bitte!", True),
        ("Gerne.", True),
        ("Okay, machen Sie das", True),
        ("Nein danke", False),
        ("nein", False),
        ("Lieber nicht.", False),
        ("Ja, aber erst nächste Woche", None),  # hedged: the LLM router decides
        ("Ja bitte, und wo ist eigentlich meine andere Bestellung NE-100231?", None),
        ("Wie lange dauert das?", None),
    ],
)
def test_bare_answers_to_the_ticket_question_need_no_llm(text: str, expected: bool | None) -> None:
    result = route_by_rules(convo(text), ASKED)

    if expected is None:
        assert result is None
    else:
        assert result is not None and result.source is RoutingSource.RULE
        assert result.confirmation is expected and result.intent is Intent.ORDER_CHANGE


def test_yes_without_a_pending_question_is_not_a_confirmation() -> None:
    assert route_by_rules(convo("Ja"), SessionState()) is None


def test_router_flags_for_change_requests_and_confirmations_are_parsed() -> None:
    text = (
        '{"intent": "HUMAN_AGENT", "confidence": 0.9, "change_request": true, '
        '"confirmation": true, "standalone_query": "Adresse ändern"}'
    )

    result = parse_llm_routing(text, convo("Ja, bitte machen Sie das für mich."))

    assert result.change_request is True and result.confirmation is True
    assert parse_llm_routing(
        '{"intent": "RETURN_REQUEST", "defect_reported": true}', convo("x")
    ).defect_reported
    assert parse_llm_routing('{"intent": "SHIPPING_INFO"}', convo("x")).confirmation is None
