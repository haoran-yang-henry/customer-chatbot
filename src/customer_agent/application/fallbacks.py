"""German fallback replies, chosen by reason code. Guardrails judge; this module decides wording."""

from customer_agent.domain.conversation import AgentResponse, Behavior
from customer_agent.domain.intents import Intent

# Contact facts duplicated from the knowledge base (info-customer-service-contact); a data test
# fails if they drift apart (tests/data/test_knowledge_base.py).
CONTACT_PHONE = "0800 123 4567"
CONTACT_EMAIL = "service@nordelektro.example"
CONTACT_HOURS = "Montag bis Freitag 8–20 Uhr, Samstag 9–16 Uhr"
CONTACT_SOURCE = "info-customer-service-contact"

# The agent never offers to hand over (ADR-0014): when it cannot help, it names the channel.
_EMAIL = f"Unser Service-Team erreichen Sie per E-Mail an {CONTACT_EMAIL}."
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
        f"Kürze erneut. {_EMAIL}",
    ),
}

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
        f"Für Fragen zu einer Erstattung erreichen Sie unser Service-Team per E-Mail an "
        f"{CONTACT_EMAIL}.",
    ),
    "ask_order_id": (
        Behavior.CLARIFY,
        "Gern helfe ich Ihnen weiter. Um welche Bestellung geht es? Bitte nennen Sie mir die "
        "Bestellnummer aus Ihrer Bestellbestätigung (sie beginnt mit „NE-“).",
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
    "ticket_declined": (
        Behavior.ANSWER,
        "In Ordnung, dann gebe ich nichts weiter. Kann ich Ihnen sonst noch helfen?",
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
    f"Dazu liegen mir leider keine gesicherten Informationen vor. {_EMAIL}",
)


def fallback(reasons: list[str]) -> AgentResponse:
    """First reason with a dedicated reply wins; anything else is an honest 'no answer'."""
    behavior, text = next((_BY_REASON[r] for r in reasons if r in _BY_REASON), _NO_ANSWER)
    return AgentResponse(answer=text, behavior=behavior, reasons=reasons)


# --- tickets (ADR-0014) -----------------------------------------------------------------------

# Every confirmation question ends like this; a stateless call recognises it in the transcript.
TICKET_QUESTION = "an unser Service-Team weitergeben?"
# Which request a question was about, so that its confirmation can be attributed to it.
_QUESTION_SUBJECTS: dict[Intent, str] = {
    Intent.ORDER_CHANGE: "Ihren Änderungswunsch",
    Intent.ORDER_CANCEL: "die Stornierung",
    Intent.RETURN_REQUEST: "die Rücksendung",
}


def ticket_question(intent: Intent | None, order_id: str | None) -> str:
    subject = _QUESTION_SUBJECTS.get(intent, "Ihr Anliegen") if intent else "Ihr Anliegen"
    about = f" zu {order_id}" if order_id else ""
    return f"Soll ich {subject}{about} {TICKET_QUESTION}"


def intent_of_question(question: str) -> Intent | None:
    return next((i for i, subject in _QUESTION_SUBJECTS.items() if subject in question), None)


def confirm_ticket_reply(intent: Intent | None, order_id: str | None) -> AgentResponse:
    """A change only the service team can make: ask once before a ticket is opened."""
    return AgentResponse(
        answer=(
            "Diese Änderung kann ich nicht selbst vornehmen, aber unser Service-Team kann sie "
            f"prüfen. {ticket_question(intent, order_id)}"
        ),
        behavior=Behavior.CLARIFY,
        intent=intent,
    )


def ticket_created_reply(ticket_id: str, *, intent: Intent | None) -> AgentResponse:
    """The confirmed request is with the team; the customer can refer to the number."""
    return AgentResponse(
        answer=(
            f"Vielen Dank. Ich habe Ihr Anliegen an unser Service-Team weitergegeben "
            f"(Ticketnummer {ticket_id}). Das Team bearbeitet es und meldet sich bei Ihnen."
        ),
        behavior=Behavior.ESCALATE,
        intent=intent,
        tool_calls=["create_ticket"],
    )


def ticket_exists_reply(ticket_id: str, *, intent: Intent | None) -> AgentResponse:
    """One ticket per conversation (and the daily limit, T12): point to the open one."""
    return AgentResponse(
        answer=(
            f"Ihr Anliegen liegt unserem Service-Team bereits vor (Ticketnummer {ticket_id}). "
            "Weitere Wünsche nennen Sie bitte dem Team, wenn es sich bei Ihnen meldet, oder per "
            f"E-Mail an {CONTACT_EMAIL}."
        ),
        behavior=Behavior.ESCALATE,
        intent=intent,
    )


_OUTAGE = "Entschuldigung, ich kann Ihre Anfrage gerade nicht automatisch bearbeiten."


def outage_contact_reply() -> AgentResponse:
    """The model provider is down (NFR-11): nothing can be understood, so nothing is forwarded;
    the customer gets the contact channels at once."""
    return AgentResponse(
        answer=(
            f"{_OUTAGE} Unsere Mitarbeiter erreichen Sie per E-Mail an {CONTACT_EMAIL} oder "
            f"telefonisch unter {CONTACT_PHONE} ({CONTACT_HOURS})."
        ),
        behavior=Behavior.ESCALATE,
        sources=[CONTACT_SOURCE],
        reasons=["provider_unavailable"],
    )


PAYMENT_DATA_NOTICE = (
    "Hinweis: Bitte senden Sie keine Bank- oder Kartendaten und keine Zugangsdaten über den Chat. "
    "Ich habe sie aus Ihrer Nachricht entfernt; sie werden nicht gespeichert."
)


def with_payment_data_notice(response: AgentResponse) -> AgentResponse:
    return response.model_copy(update={"answer": f"{PAYMENT_DATA_NOTICE}\n\n{response.answer}"})


def contact_reply(*, intent: Intent | None, change_request: bool = False) -> AgentResponse:
    """How to reach a person, without a ticket: nothing needs changing, or the customer is
    anonymous (a ticket could not be answered without verified contact details)."""
    opening = "Das tut mir sehr leid. " if intent is Intent.COMPLAINT else ""
    login = (
        " Änderungswünsche kann ich an das Team weitergeben, wenn Sie in Ihrem Kundenkonto "
        "angemeldet sind."
        if change_request
        else ""
    )
    return AgentResponse(
        answer=(
            f"{opening}Unser Service-Team erreichen Sie per E-Mail an {CONTACT_EMAIL} oder "
            f"telefonisch unter {CONTACT_PHONE} ({CONTACT_HOURS}).{login}"
        ),
        behavior=Behavior.ESCALATE,
        intent=intent,
        sources=[CONTACT_SOURCE],
    )
