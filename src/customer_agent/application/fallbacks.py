"""German fallback replies, chosen by reason code. Guardrails judge; this module decides wording."""

from customer_agent.domain.conversation import AgentResponse, Behavior

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

_NO_ANSWER = (
    Behavior.NO_ANSWER,
    f"Dazu liegen mir leider keine gesicherten Informationen vor. {_HANDOFF}",
)


def fallback(reasons: list[str]) -> AgentResponse:
    """First reason with a dedicated reply wins; anything else is an honest 'no answer'."""
    behavior, text = next((_BY_REASON[r] for r in reasons if r in _BY_REASON), _NO_ANSWER)
    return AgentResponse(answer=text, behavior=behavior, reasons=reasons)
