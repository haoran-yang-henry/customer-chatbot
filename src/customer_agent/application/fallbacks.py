"""German fallback replies, chosen by reason code. Guardrails judge; this module decides wording."""

from customer_agent.domain.conversation import AgentResponse, Behavior
from customer_agent.domain.intents import Intent

_HANDOFF = "Gern verbinde ich Sie mit einem Mitarbeiter unseres Kundenservice."

_BY_REASON: dict[str, tuple[Behavior, str]] = {
    "empty_input": (Behavior.CLARIFY, "Wie kann ich Ihnen helfen?"),
    "input_too_long": (
        Behavior.CLARIFY,
        "Ihre Nachricht ist leider zu lang. Bitte fassen Sie Ihr Anliegen in wenigen Sätzen "
        "zusammen.",
    ),
    "provider_unavailable": (
        Behavior.NO_ANSWER,
        "Entschuldigung, ich kann Ihre Anfrage gerade nicht bearbeiten. Bitte versuchen Sie es in "
        f"Kürze erneut. {_HANDOFF}",
    ),
}

_STEP_REPLIES: dict[str, tuple[Behavior, str]] = {
    # The greeting discloses that the customer talks to an AI (EU AI Act, transparency).
    "greet": (
        Behavior.ANSWER,
        "Guten Tag! Ich bin der digitale Assistent von NordElektro. Wie kann ich Ihnen helfen?",
    ),
    "refuse_out_of_scope": (
        Behavior.REFUSE,
        "Dabei kann ich Ihnen leider nicht helfen. Gern beantworte ich Fragen zu Bestellungen, "
        "Versand, Rücksendungen, Zahlung, Garantie und unseren Produkten.",
    ),
    "ask_order_id": (
        Behavior.CLARIFY,
        "Gern helfe ich Ihnen weiter. Um welche Bestellung geht es? Bitte nennen Sie mir Ihre "
        "Bestellnummer (zum Beispiel NE-123456).",
    ),
    "ask_login": (
        Behavior.REFUSE,
        "Zu einzelnen Bestellungen kann ich Ihnen nur Auskunft geben, wenn Sie in Ihrem "
        "Kundenkonto angemeldet sind. Bitte melden Sie sich an und schreiben Sie mir dann erneut.",
    ),
    "handoff": (
        Behavior.ESCALATE,
        "Ich leite Ihr Anliegen gern an einen Mitarbeiter unseres Kundenservice weiter. Sie "
        "erhalten eine Rückmeldung, sobald sich ein Kollege Ihr Anliegen angesehen hat.",
    ),
    "handoff_complaint": (
        Behavior.ESCALATE,
        "Das tut mir sehr leid. Ich leite Ihr Anliegen an einen Mitarbeiter unseres "
        "Kundenservice weiter, der sich persönlich darum kümmert. Sie erhalten innerhalb von "
        "2 Werktagen eine Rückmeldung.",
    ),
    "clarify": (
        Behavior.CLARIFY,
        "Könnten Sie Ihr Anliegen bitte etwas genauer beschreiben? Dann helfe ich Ihnen gern "
        "weiter.",
    ),
}


def step_reply(
    step: str, *, intent: Intent | None = None, text: str | None = None
) -> AgentResponse:
    """Template reply for a state-machine step; ``text`` overrides the wording (e.g. a
    router-generated clarification question)."""
    behavior, default = _STEP_REPLIES[step]
    return AgentResponse(answer=text or default, behavior=behavior, intent=intent)


_NO_ANSWER = (
    Behavior.NO_ANSWER,
    f"Dazu liegen mir leider keine gesicherten Informationen vor. {_HANDOFF}",
)


def fallback(reasons: list[str]) -> AgentResponse:
    """First reason with a dedicated reply wins; anything else is an honest 'no answer'."""
    behavior, text = next((_BY_REASON[r] for r in reasons if r in _BY_REASON), _NO_ANSWER)
    return AgentResponse(answer=text, behavior=behavior, reasons=reasons)
