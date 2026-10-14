"""Input guardrail: one pure function from the raw message to a ``GuardrailDecision``.

Order: normalise (NFKC, invisible characters out) → size → prompt-injection signals → personal
data masked. The returned ``content`` is the only version of the message that the router, the
answer model, the transcript and the logs ever see.
"""

from customer_agent.application.guardrails.patterns import (
    BLOCK_THRESHOLD,
    injection_score,
    injection_signals,
    normalize,
)
from customer_agent.domain.guardrails import GuardrailAction, GuardrailDecision
from customer_agent.domain.pii import mask_personal_data

MAX_INPUT_CHARS = 2000

# Reasons after which the customer is told that data was removed from their message.
PAYMENT_DATA_REASONS = frozenset({"pii_iban", "pii_payment_card", "secret"})


def check_input(message: str) -> GuardrailDecision:
    clean, hidden, removed = normalize(message)
    reasons = ["invisible_characters"] if removed else []
    if not clean:
        return GuardrailDecision(
            stage="input", action=GuardrailAction.BLOCK, reasons=["empty_input"], content=""
        )
    if len(clean) > MAX_INPUT_CHARS:
        masked, _ = mask_personal_data(clean[:MAX_INPUT_CHARS])
        return GuardrailDecision(
            stage="input",
            action=GuardrailAction.BLOCK,
            reasons=["input_too_long"],
            content=masked,
        )
    # Hidden text is scored too: tag characters only exist to smuggle instructions.
    hits = injection_signals(f"{clean}\n{hidden}" if hidden else clean)
    score = injection_score(hits)
    masked, pii = mask_personal_data(clean)
    content = masked if masked != message else None
    if score >= BLOCK_THRESHOLD:
        return GuardrailDecision(
            stage="input",
            action=GuardrailAction.BLOCK,
            reasons=["prompt_injection", *(f"signal:{name}" for name in sorted(hits)), *reasons],
            content=masked,
            scores={"injection": score},
        )
    reasons += pii
    action = GuardrailAction.REDACT if reasons else GuardrailAction.ALLOW
    scores = {"injection": score} if score else {}
    return GuardrailDecision(
        stage="input", action=action, reasons=reasons, content=content, scores=scores
    )
