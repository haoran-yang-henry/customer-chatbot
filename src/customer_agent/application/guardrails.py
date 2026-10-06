"""Day 3 guardrails: input size and output citation/length checks. Day 6 adds injection, PII,
scope and language checks behind the same ``GuardrailDecision`` contract."""

from customer_agent.domain.guardrails import GuardrailAction, GuardrailDecision

MAX_INPUT_CHARS = 2000
MAX_ANSWER_CHARS = 2000


def check_input(message: str) -> GuardrailDecision:
    text = message.strip()
    if not text:
        return GuardrailDecision(
            stage="input", action=GuardrailAction.BLOCK, reasons=["empty_input"]
        )
    if len(text) > MAX_INPUT_CHARS:
        return GuardrailDecision(
            stage="input", action=GuardrailAction.BLOCK, reasons=["input_too_long"]
        )
    return GuardrailDecision(stage="input", action=GuardrailAction.ALLOW)


def check_output(answer: str, cited: list[str], offered: set[str]) -> GuardrailDecision:
    """Every knowledge answer must cite at least one document it was actually given (FR-11)."""
    reasons = []
    if not set(cited) & offered:
        reasons.append("missing_citation")
    if len(answer) > MAX_ANSWER_CHARS:
        reasons.append("answer_too_long")
    action = GuardrailAction.BLOCK if reasons else GuardrailAction.ALLOW
    return GuardrailDecision(stage="output", action=action, reasons=reasons)
