from customer_agent.application.fallbacks import (
    PAYMENT_DATA_NOTICE,
    fallback,
    handoff_pending_reply,
    step_reply,
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
    assert "Mitarbeiter" in unknown.answer


def test_attacks_get_the_ordinary_out_of_scope_refusal() -> None:
    """A refusal must not tell an attacker which detector fired."""
    out_of_scope = step_reply("refuse_out_of_scope").answer

    for reason in ("prompt_injection", "prompt_injection_llm"):
        reply = fallback([reason, "signal:override"])
        assert reply.behavior is Behavior.REFUSE and reply.answer == out_of_scope


def test_money_demand_refusal_promises_nothing_and_offers_a_human() -> None:
    reply = step_reply("refuse_money_demand")

    assert reply.behavior is Behavior.REFUSE
    assert "weder veranlassen noch zusagen" in reply.answer and "Mitarbeiter" in reply.answer


def test_ticket_limit_points_to_the_open_ticket_without_a_new_one() -> None:
    reply = handoff_pending_reply("T-000007", intent=Intent.HUMAN_AGENT)

    assert reply.behavior is Behavior.ESCALATE and "T-000007" in reply.answer
    assert reply.tool_calls == []


def test_payment_data_notice_comes_first() -> None:
    reply = with_payment_data_notice(AgentResponse(answer="Antwort.", behavior=Behavior.ANSWER))

    assert reply.answer == f"{PAYMENT_DATA_NOTICE}\n\nAntwort."
