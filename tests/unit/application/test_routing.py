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
    for text in ["Sorry, I cannot.", '{"intent": "ORDER_CHANGE"}', "{broken"]:
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
