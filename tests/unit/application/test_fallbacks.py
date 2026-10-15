import pytest

from customer_agent.application.fallbacks import (
    CONTACT_EMAIL,
    PAYMENT_DATA_NOTICE,
    TICKET_QUESTION,
    contact_reply,
    fallback,
    intent_of_question,
    step_reply,
    ticket_exists_reply,
    ticket_question,
    with_payment_data_notice,
)
from customer_agent.domain.conversation import AgentResponse, Behavior
from customer_agent.domain.intents import Intent


def test_fallback_wording_depends_on_reason() -> None:
    assert fallback(["empty_input"]).behavior is Behavior.CLARIFY
    assert fallback(["input_too_long"]).answer.startswith("Ihre Nachricht ist leider zu lang")
    unknown = fallback(["missing_citation"])
    assert unknown.behavior is Behavior.NO_ANSWER
    assert unknown.reasons == ["missing_citation"]
    assert CONTACT_EMAIL in unknown.answer and "verbinde" not in unknown.answer  # ADR-0014


def test_attacks_get_the_ordinary_out_of_scope_refusal() -> None:
    """A refusal must not tell an attacker which detector fired."""
    out_of_scope = step_reply("refuse_out_of_scope").answer

    for reason in ("prompt_injection", "prompt_injection_llm"):
        reply = fallback([reason, "signal:override"])
        assert reply.behavior is Behavior.REFUSE and reply.answer == out_of_scope


def test_money_demand_refusal_promises_nothing_and_names_the_channel() -> None:
    reply = step_reply("refuse_money_demand")

    assert reply.behavior is Behavior.REFUSE
    assert "weder veranlassen noch zusagen" in reply.answer and CONTACT_EMAIL in reply.answer
    assert "verbinde" not in reply.answer


def test_an_open_ticket_is_pointed_to_without_a_new_one() -> None:
    reply = ticket_exists_reply("T-000007", intent=Intent.HUMAN_AGENT)

    assert reply.behavior is Behavior.ESCALATE and "T-000007" in reply.answer
    assert reply.tool_calls == []


@pytest.mark.parametrize(
    ("intent", "order_id"),
    [
        (Intent.ORDER_CHANGE, "NE-100260"),
        (Intent.ORDER_CANCEL, "NE-100245"),
        (Intent.RETURN_REQUEST, "NE-100231"),
        (None, None),
    ],
)
def test_a_ticket_question_names_what_would_be_forwarded(
    intent: Intent | None, order_id: str | None
) -> None:
    question = ticket_question(intent, order_id)

    assert question.endswith(TICKET_QUESTION) and question.startswith("Soll ich")
    assert intent_of_question(question) is intent  # a stateless call can attribute the answer
    assert (order_id or "Ihr Anliegen") in question


def test_a_named_question_still_belongs_to_its_intent() -> None:
    question = ticket_question(Intent.RETURN_REQUEST, "NE-100198", subject="Ihre Defektmeldung")

    assert question == f"Soll ich Ihre Defektmeldung zu NE-100198 {TICKET_QUESTION}"
    assert intent_of_question(question) is Intent.RETURN_REQUEST


def test_contact_details_mention_the_login_only_for_change_requests() -> None:
    plain = contact_reply(intent=Intent.HUMAN_AGENT)
    change = contact_reply(intent=Intent.HUMAN_AGENT, change_request=True)

    assert CONTACT_EMAIL in plain.answer and "angemeldet" not in plain.answer
    assert "angemeldet" in change.answer
    assert plain.tool_calls == [] and plain.sources == ["info-customer-service-contact"]


def test_payment_data_notice_comes_first() -> None:
    reply = with_payment_data_notice(AgentResponse(answer="Antwort.", behavior=Behavior.ANSWER))

    assert reply.answer == f"{PAYMENT_DATA_NOTICE}\n\nAntwort."
