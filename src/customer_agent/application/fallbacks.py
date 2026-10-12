"""German fallback replies, chosen by reason code. Guardrails judge; this module decides wording."""

from customer_agent.domain.conversation import AgentResponse, Behavior
from customer_agent.domain.intents import Intent

_HANDOFF = "Gern verbinde ich Sie mit einem Mitarbeiter unseres Kundenservice."
_SUPPORTED_TOPICS = (
    "Gern beantworte ich Fragen zu Bestellungen, Versand, Rücksendungen, Zahlung, Garantie und "
    "unseren Produkten."
)
# Same wording as any out-of-scope request: a refusal must not tell an attacker which detector
# fired or how close the attempt came.
_REFUSE = f"Dabei kann ich Ihnen leider nicht helfen. {_SUPPORTED_TOPICS}"

_BY_REASON: dict[str, tuple[Behavior, str]] = {
    "prompt_injection": (Behavior.REFUSE, _REFUSE),
    "prompt_injection_llm": (Behavior.REFUSE, _REFUSE),
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

# Contact facts duplicated from the knowledge base (info-customer-service-contact); a data test
# fails if they drift apart (tests/data/test_knowledge_base.py).
CONTACT_PHONE = "0800 123 4567"
CONTACT_EMAIL = "service@nordelektro.example"
CONTACT_HOURS = "Montag bis Freitag 8–20 Uhr, Samstag 9–16 Uhr"
CONTACT_SOURCE = "info-customer-service-contact"

_STEP_REPLIES: dict[str, tuple[Behavior, str]] = {
    # The greeting discloses that the customer talks to an AI (EU AI Act, transparency).
    "greet": (
        Behavior.ANSWER,
        "Guten Tag! Ich bin der digitale Assistent von NordElektro. Wie kann ich Ihnen helfen?",
    ),
    "refuse_out_of_scope": (Behavior.REFUSE, _REFUSE),
    # Money cannot be granted by the agent (ADR-0006), whatever is claimed (S-007, T07).
    "refuse_money_demand": (
        Behavior.REFUSE,
        "Erstattungen, Gutschriften, Gutscheine oder Rabatte kann ich leider weder veranlassen "
        "noch zusagen – auch nicht auf Grundlage von Absprachen, die ich nicht überprüfen kann. "
        "Wenn Sie möchten, verbinde ich Sie gern mit einem Mitarbeiter, der Ihr Anliegen prüft.",
    ),
    "ask_order_id": (
        Behavior.CLARIFY,
        "Gern helfe ich Ihnen weiter. Um welche Bestellung geht es? Bitte nennen Sie mir Ihre "
        "Bestellnummer (zum Beispiel NE-123456).",
    ),
    "refuse_third_party": (
        Behavior.REFUSE,
        "Aus Datenschutzgründen kann ich nur Auskunft zu Bestellungen aus Ihrem eigenen "
        "Kundenkonto geben. Die betreffende Person kann sich selbst an uns wenden.",
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


def handoff_reply(ticket_id: str, *, intent: Intent | None) -> AgentResponse:
    """Handoff with the ticket number the customer can refer to (Day 5)."""
    key = "handoff_complaint" if intent is Intent.COMPLAINT else "handoff"
    behavior, text = _STEP_REPLIES[key]
    return AgentResponse(
        answer=f"{text} Ihre Ticketnummer lautet {ticket_id}.",
        behavior=behavior,
        intent=intent,
        tool_calls=["create_ticket"],
    )


def handoff_pending_reply(ticket_id: str, *, intent: Intent | None) -> AgentResponse:
    """The customer reached the ticket limit (Day 6): point to the open ticket instead of
    opening another one, so a loop or a script cannot flood the service team."""
    return AgentResponse(
        answer=(
            f"Ihr Anliegen liegt unseren Mitarbeitern bereits vor (Ticketnummer {ticket_id}). "
            "Sie erhalten eine Rückmeldung, sobald sich ein Kollege darum gekümmert hat."
        ),
        behavior=Behavior.ESCALATE,
        intent=intent,
    )


PAYMENT_DATA_NOTICE = (
    "Hinweis: Bitte senden Sie keine Bank- oder Kartendaten und keine Zugangsdaten über den Chat. "
    "Ich habe sie aus Ihrer Nachricht entfernt; sie werden nicht gespeichert."
)


def with_payment_data_notice(response: AgentResponse) -> AgentResponse:
    return response.model_copy(update={"answer": f"{PAYMENT_DATA_NOTICE}\n\n{response.answer}"})


def contact_reply(*, intent: Intent | None) -> AgentResponse:
    """Handoff for anonymous customers: how to reach a human, no ticket (it could not be
    answered without contact details, and the chat does not collect them)."""
    opening = "Das tut mir sehr leid. " if intent is Intent.COMPLAINT else ""
    return AgentResponse(
        answer=(
            f"{opening}Unsere Mitarbeiter erreichen Sie telefonisch unter {CONTACT_PHONE} "
            f"({CONTACT_HOURS}) oder per E-Mail an {CONTACT_EMAIL}. Wenn Sie sich in Ihrem "
            "Kundenkonto anmelden, kann ich Ihr Anliegen auch direkt an einen Kollegen übergeben."
        ),
        behavior=Behavior.ESCALATE,
        intent=intent,
        sources=[CONTACT_SOURCE],
    )
