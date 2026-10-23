"""Input guardrail: pure steps from the raw message to a ``GuardrailDecision``.

Order: normalise (NFKC, invisible characters out) → size → prompt-injection signals → personal
data masked. ``check_input`` runs all steps at once; the understand agent's middleware runs them
one hook each (rebuild phase 2), so both take the same path. The decision's ``content`` is the
only version of the message that the router, the answer model, the transcript and the logs ever
see.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

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


@dataclass(frozen=True)
class InputScreen:
    """The check so far: the message text (normalised, then masked), the invisible text it
    carried, and the verdict. Built from the raw message; never keeps it."""

    text: str
    hidden: str = ""
    reasons: tuple[str, ...] = ()
    blocked: bool = False
    scores: Mapping[str, float] = field(default_factory=dict)


def sanitize(message: str) -> InputScreen:
    """Normalise; block an empty or over-long message (the latter is cut, masked later)."""
    clean, hidden, removed = normalize(message)
    if not clean:
        return InputScreen(text="", reasons=("empty_input",), blocked=True)
    if len(clean) > MAX_INPUT_CHARS:
        return InputScreen(text=clean[:MAX_INPUT_CHARS], reasons=("input_too_long",), blocked=True)
    return InputScreen(
        text=clean, hidden=hidden, reasons=("invisible_characters",) if removed else ()
    )


def screen_injection(screen: InputScreen) -> InputScreen:
    """Score the unmasked text; hidden text is scored too: tag characters only exist to smuggle
    instructions."""
    if screen.blocked:
        return screen
    hits = injection_signals(f"{screen.text}\n{screen.hidden}" if screen.hidden else screen.text)
    score = injection_score(hits)
    if score >= BLOCK_THRESHOLD:
        signals = tuple(f"signal:{name}" for name in sorted(hits))
        return replace(
            screen,
            reasons=("prompt_injection", *signals, *screen.reasons),
            blocked=True,
            scores={"injection": score},
        )
    return replace(screen, scores={"injection": score} if score else {})


def mask_input(screen: InputScreen) -> InputScreen:
    """Mask personal data; a blocked message is masked as well, without adding reasons."""
    masked, found = mask_personal_data(screen.text)
    if screen.blocked:
        return replace(screen, text=masked)
    return replace(screen, text=masked, reasons=(*screen.reasons, *found))


def input_decision(screen: InputScreen, original: str | None = None) -> GuardrailDecision:
    """``content`` is the masked text; ``None`` when it equals ``original`` (nothing changed)."""
    content: str | None
    if screen.blocked:
        action, content = GuardrailAction.BLOCK, screen.text
    else:
        action = GuardrailAction.REDACT if screen.reasons else GuardrailAction.ALLOW
        content = None if screen.text == original else screen.text
    return GuardrailDecision(
        stage="input",
        action=action,
        reasons=list(screen.reasons),
        content=content,
        scores=dict(screen.scores),
    )


def check_input(message: str) -> GuardrailDecision:
    return input_decision(mask_input(screen_injection(sanitize(message))), original=message)
