"""Order-specific guidance (Day 5, ADR-0006): facts from the order, eligibility from domain rules
and self-service steps with a link. Deterministic templates, no LLM: order facts (status, dates,
amounts) must be exact. When a change is still possible, ``Guidance.change_possible`` lets the
orchestrator ask whether to forward it to the service team (ADR-0014).
"""

from dataclasses import dataclass
from datetime import date

from customer_agent.application.fallbacks import CONTACT_EMAIL
from customer_agent.application.formatting import format_date_de, format_eur
from customer_agent.domain.conversation import AgentResponse, Behavior
from customer_agent.domain.intents import Intent
from customer_agent.domain.orders import (
    PRESUMPTION_YEARS,
    Order,
    OrderStatus,
    can_cancel,
    can_change_address,
    can_claim_warranty,
    can_return,
    defect_presumed,
)

ACCOUNT_URL = "https://www.nordelektro.example/konto/bestellungen"  # as in the knowledge base
TOOL = "get_order_status"


@dataclass(frozen=True)
class Guidance:
    response: AgentResponse
    change_possible: bool = False  # the customer's change can still be made: offer a ticket
    question_subject: str | None = None  # what the ticket question names, if not the intent's


def _reply(text: str, behavior: Behavior, intent: Intent, sources: list[str]) -> AgentResponse:
    return AgentResponse(
        answer=text, behavior=behavior, intent=intent, sources=sources, tool_calls=[TOOL]
    )


def order_not_found(order_id: str, intent: Intent) -> AgentResponse:
    """Same wording whether the order does not exist or belongs to someone else (T06)."""
    return _reply(
        f"Ich konnte in Ihrem Kundenkonto keine Bestellung {order_id} finden. Bitte prüfen Sie "
        "die Bestellnummer – Sie finden sie in Ihrer Bestellbestätigung.",
        Behavior.REFUSE,
        intent,
        [],
    )


def _items(order: Order) -> str:
    return ", ".join(
        item.name if item.quantity == 1 else f"{item.quantity} × {item.name}"
        for item in order.items
    )


def _status(order: Order) -> tuple[str, list[str]]:
    shipping, oid = order.shipping, order.order_id
    expected = (
        f" Voraussichtliche Lieferung: {format_date_de(shipping.expected_delivery)}."
        if shipping.expected_delivery
        else ""
    )
    match order.status:
        case OrderStatus.PENDING_PAYMENT:
            failed = (
                " – die letzte Zahlung ist fehlgeschlagen"
                if order.payment_status == "failed"
                else ""
            )
            return (
                f"Für Ihre Bestellung {oid} ist die Zahlung noch ausstehend{failed}. Die "
                "Bestellung bleibt 48 Stunden reserviert. Unter „Meine Bestellungen“ → „Erneut "
                f"bezahlen“ können Sie eine andere Zahlungsart wählen: {ACCOUNT_URL}",
                ["faq-payment-failed"],
            )
        case OrderStatus.PAID:
            return f"Ihre Bestellung {oid} ist bezahlt und wird in Kürze bearbeitet.{expected}", []
        case OrderStatus.PROCESSING:
            return f"Ihre Bestellung {oid} wird gerade für den Versand vorbereitet.{expected}", []
        case OrderStatus.SHIPPED:
            shipped = format_date_de(shipping.shipped_at) if shipping.shipped_at else "kürzlich"
            tracking = (
                f" Sendungsnummer: {shipping.tracking_number}." if shipping.tracking_number else ""
            )
            arrival = expected.replace("Lieferung", "Zustellung")
            return (
                f"Ihre Bestellung {oid} wurde am {shipped} mit {shipping.carrier} versendet "
                f"und ist unterwegs.{arrival}{tracking}",
                [],
            )
        case OrderStatus.DELIVERED:
            delivered = format_date_de(shipping.delivered_at) if shipping.delivered_at else ""
            return f"Ihre Bestellung {oid} wurde am {delivered} zugestellt.", []
        case OrderStatus.CANCELLED:
            return f"Ihre Bestellung {oid} wurde storniert.", []
        case OrderStatus.RETURN_REQUESTED | OrderStatus.RETURNED:
            return _refund(order)
    raise AssertionError(f"unhandled status {order.status}")  # pragma: no cover


def _refund(order: Order) -> tuple[str, list[str]]:
    oid, info = order.order_id, order.return_info
    if info is None:
        return f"Für Ihre Bestellung {oid} ist keine Rücksendung erfasst.", []
    if info.refund_status == "refunded" and info.refunded_at:
        amount = f" über {format_eur(info.refund_amount)}" if info.refund_amount else ""
        return (
            f"Die Erstattung{amount} für Ihre Rücksendung zu {oid} wurde am "
            f"{format_date_de(info.refunded_at)} veranlasst.",
            ["faq-refund-timeline"],
        )
    stage = "ist bei uns eingegangen" if info.received_at else "ist auf dem Weg zu uns"
    return (
        f"Ihre Rücksendung zu {oid} {stage}. Wir erstatten den Kaufpreis innerhalb von 14 Tagen "
        "nach Eingang der Rücksendung in unserem Lager über die ursprüngliche Zahlungsart. Den "
        f"aktuellen Stand sehen Sie unter „Meine Bestellungen“: {ACCOUNT_URL}",
        ["faq-refund-timeline"],
    )


def _cancel(order: Order) -> Guidance:
    oid, eligibility = order.order_id, can_cancel(order)
    sources = ["policy-order-cancellation"]
    if eligibility.allowed:
        text = (
            f"Ihre Bestellung {oid} ({_items(order)}, {format_eur(order.total)}) ist noch nicht "
            "versendet und kann kostenlos storniert werden. So geht's: Kundenkonto → „Meine "
            f"Bestellungen“ → „Stornieren“: {ACCOUNT_URL} – bereits gezahlte Beträge erstatten "
            "wir innerhalb von 5 Werktagen."
        )
        return Guidance(_reply(text, Behavior.ANSWER, Intent.ORDER_CANCEL, sources), True)
    elif eligibility.reason == "already_cancelled":
        text = f"Ihre Bestellung {oid} ist bereits storniert."
    else:
        text = (
            f"Ihre Bestellung {oid} wurde bereits versendet und kann daher nicht mehr storniert "
            "werden. Sie können die Annahme des Pakets verweigern oder den Artikel nach Erhalt "
            "innerhalb von 30 Tagen zurücksenden."
        )
    return Guidance(_reply(text, Behavior.ANSWER, Intent.ORDER_CANCEL, sources))


def _return(order: Order, today: date, *, defect: bool) -> Guidance:
    oid, eligibility = order.order_id, can_return(order, today)
    if defect and eligibility.reason == "window_expired":
        return _warranty(order, today)  # the return window is over, the warranty is not
    shipping = order.shipping
    if eligibility.allowed and eligibility.deadline and shipping.delivered_at:
        seal = (
            " Bitte beachten Sie: In-Ear-Kopfhörer können aus Hygienegründen nur mit "
            "unbeschädigtem Hygienesiegel zurückgegeben werden."
            if order.needs_hygiene_seal
            else ""
        )
        text = (
            f"Ihre Bestellung {oid} wurde am {format_date_de(shipping.delivered_at)} zugestellt; "
            f"Sie können sie bis zum {format_date_de(eligibility.deadline)} zurücksenden. So "
            "geht's: Kundenkonto → „Meine Bestellungen“ → „Artikel zurücksenden“, dann das "
            "Retourenlabel ausdrucken oder den QR-Code in einer DHL-Filiale vorzeigen: "
            f"{ACCOUNT_URL}. Die Rücksendung innerhalb Deutschlands ist kostenlos, aus Österreich "
            f"berechnen wir 6,95 €.{seal}"
        )
        return Guidance(
            _reply(text, Behavior.ANSWER, Intent.RETURN_REQUEST, ["policy-returns-v5"]), True
        )
    if eligibility.reason == "window_expired" and eligibility.deadline:
        text = (
            f"Die 30-tägige Rückgabefrist für Ihre Bestellung {oid} ist am "
            f"{format_date_de(eligibility.deadline)} abgelaufen. Ist der Artikel defekt, gilt die "
            "gesetzliche Gewährleistung von 2 Jahren ab Zustellung: Kundenkonto → „Meine "
            f"Bestellungen“ → „Problem melden“: {ACCOUNT_URL}."
        )
        sources = ["policy-returns-v5", "policy-warranty"]
        return Guidance(_reply(text, Behavior.ANSWER, Intent.RETURN_REQUEST, sources))
    messages = {
        "return_in_progress": f"Für Ihre Bestellung {oid} ist bereits eine Rücksendung angemeldet.",
        "already_returned": f"Ihre Bestellung {oid} wurde bereits zurückgesendet.",
        "cancelled": f"Ihre Bestellung {oid} wurde storniert; eine Rücksendung ist nicht nötig.",
        "not_delivered": (
            f"Ihre Bestellung {oid} ist noch nicht zugestellt. Eine Rücksendung ist ab der "
            "Zustellung möglich; solange sie nicht versendet ist, können Sie sie stornieren."
        ),
    }
    text = messages.get(
        eligibility.reason,
        f"Eine Rücksendung zu Ihrer Bestellung {oid} kann ich leider nicht prüfen. Unser "
        f"Service-Team erreichen Sie per E-Mail an {CONTACT_EMAIL}.",
    )
    return Guidance(_reply(text, Behavior.ANSWER, Intent.RETURN_REQUEST, ["policy-returns-v5"]))


def _warranty(order: Order, today: date) -> Guidance:
    """policy-warranty: the customer reported a defect after the return window."""
    oid, claim = order.order_id, can_claim_warranty(order, today)
    sources = ["policy-warranty"]
    if not claim.allowed or claim.deadline is None:
        until = f" ist am {format_date_de(claim.deadline)} abgelaufen" if claim.deadline else ""
        text = (
            f"Die gesetzliche Gewährleistung für Ihre Bestellung {oid}{until}. Ob der Hersteller "
            f"eine Garantie gewährt, klärt unser Service-Team per E-Mail an {CONTACT_EMAIL}."
        )
        return Guidance(_reply(text, Behavior.ANSWER, Intent.RETURN_REQUEST, sources))
    presumption = (
        f" Da die Zustellung weniger als {12 * PRESUMPTION_YEARS} Monate zurückliegt, wird "
        "vermutet, dass der Mangel "
        "schon bei der Übergabe vorlag."
        if defect_presumed(order, today)
        else ""
    )
    text = (
        f"Für Ihre Bestellung {oid} ({_items(order)}) gilt die gesetzliche Gewährleistung noch "
        f"bis zum {format_date_de(claim.deadline)}.{presumption} Sie können zwischen Reparatur "
        "und Ersatzlieferung wählen; die Rücksendung der defekten Ware ist für Sie kostenlos. So "
        "melden Sie den Defekt: Kundenkonto → „Meine Bestellungen“ → „Problem melden“: "
        f"{ACCOUNT_URL}."
    )
    reply = _reply(text, Behavior.ANSWER, Intent.RETURN_REQUEST, sources)
    return Guidance(reply, change_possible=True, question_subject="Ihre Defektmeldung")


def _change(order: Order) -> Guidance:
    """faq-order-change. Items cannot be changed at all; the address only before processing."""
    oid, eligibility = order.order_id, can_change_address(order)
    sources = ["faq-order-change"]
    if eligibility.allowed:
        text = (
            f"Ihre Bestellung {oid} ({_items(order)}) ist noch nicht in Bearbeitung, daher kann "
            "unser Service-Team die Lieferadresse noch ändern. Artikel lassen sich nach der "
            "Bestellung nicht mehr hinzufügen oder entfernen – dafür stornieren Sie die "
            "Bestellung und bestellen neu."
        )
        return Guidance(_reply(text, Behavior.ACTION, Intent.ORDER_CHANGE, sources), True)
    shipping = order.shipping
    redirect = (
        f"Über die Sendungsverfolgung von {shipping.carrier} können Sie die Zustellung an eine "
        "Packstation oder Filiale umleiten."
    )
    texts = {
        "processing": (
            f"Ihre Bestellung {oid} wird bereits für den Versand vorbereitet; die Lieferadresse "
            "kann daher nicht mehr geändert werden. Sobald sie versendet ist, können Sie die "
            "Zustellung über die DHL-Sendungsverfolgung an eine Packstation oder Filiale umleiten."
        ),
        "shipped": (
            f"Ihre Bestellung {oid} wurde bereits versendet; die Lieferadresse kann nicht mehr "
            f"geändert werden. {redirect}"
            + (f" Sendungsnummer: {shipping.tracking_number}." if shipping.tracking_number else "")
        ),
        "cancelled": f"Ihre Bestellung {oid} wurde storniert; eine Änderung ist nicht möglich.",
        "delivered": f"Ihre Bestellung {oid} wurde bereits zugestellt; eine Änderung ist nicht "
        "mehr möglich.",
    }
    return Guidance(
        _reply(texts[eligibility.reason], Behavior.ACTION, Intent.ORDER_CHANGE, sources)
    )


def order_guidance(intent: Intent, order: Order, today: date, *, defect: bool = False) -> Guidance:
    match intent:
        case Intent.ORDER_CANCEL:
            return _cancel(order)
        case Intent.RETURN_REQUEST:
            return _return(order, today, defect=defect)
        case Intent.ORDER_CHANGE:
            return _change(order)
        case Intent.REFUND_STATUS:
            text, sources = _refund(order)
            return Guidance(_reply(text, Behavior.ACTION, intent, sources))
        case _:
            text, sources = _status(order)
            return Guidance(_reply(text, Behavior.ACTION, Intent.ORDER_STATUS, sources))
