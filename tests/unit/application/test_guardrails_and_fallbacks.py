from customer_agent.application.fallbacks import fallback
from customer_agent.application.guardrails import (
    MAX_ANSWER_CHARS,
    MAX_INPUT_CHARS,
    check_input,
    check_output,
)
from customer_agent.domain.conversation import Behavior
from customer_agent.domain.guardrails import GuardrailAction


def test_normal_input_is_allowed() -> None:
    assert check_input("Kostet die Rücksendung etwas?").action is GuardrailAction.ALLOW


def test_empty_and_oversized_input_are_blocked_with_reason() -> None:
    assert check_input("  ").reasons == ["empty_input"]
    assert check_input("x" * (MAX_INPUT_CHARS + 1)).reasons == ["input_too_long"]


def test_answer_must_cite_an_offered_document() -> None:
    assert check_output("Text", ["policy-returns-v5"], {"policy-returns-v5"}).action == "allow"
    assert check_output("Text", [], {"policy-returns-v5"}).reasons == ["missing_citation"]
    # Citing an id that was never offered (hallucinated source) is not a citation.
    assert check_output("Text", ["made-up-doc"], {"policy-returns-v5"}).blocked


def test_overlong_answer_is_blocked() -> None:
    decision = check_output("x" * (MAX_ANSWER_CHARS + 1), ["a-b"], {"a-b"})

    assert decision.reasons == ["answer_too_long"]


def test_fallback_wording_depends_on_reason() -> None:
    assert fallback(["empty_input"]).behavior is Behavior.CLARIFY
    assert fallback(["input_too_long"]).answer.startswith("Ihre Nachricht ist leider zu lang")
    unknown = fallback(["missing_citation"])
    assert unknown.behavior is Behavior.NO_ANSWER
    assert unknown.reasons == ["missing_citation"]
    assert "Mitarbeiter" in unknown.answer
