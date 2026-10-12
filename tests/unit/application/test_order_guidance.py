"""Guidance texts for every seed order (data/seed/README.md lists why each order exists)."""

from datetime import date
from decimal import Decimal

import pytest

from customer_agent.application.formatting import format_date_de, format_eur
from customer_agent.application.order_guidance import ACCOUNT_URL, order_guidance, order_not_found
from customer_agent.domain.conversation import Behavior
from customer_agent.domain.intents import Intent
from tests.fakes import SeedOrderRepository

TODAY = date(2026, 10, 5)
ORDERS = SeedOrderRepository().orders


def test_german_formats() -> None:
    assert format_date_de(date(2026, 10, 6)) == "6. Oktober 2026"
    assert format_date_de(date(2026, 3, 1)) == "1. März 2026"
    assert format_eur(Decimal("1099")) == "1.099,00 €"
    assert format_eur(Decimal("54.94")) == "54,94 €"


@pytest.mark.parametrize(
    ("order_id", "intent", "behavior", "phrases", "sources"),
    [
        (
            "NE-100250",
            Intent.ORDER_STATUS,
            Behavior.ACTION,
            ["unterwegs", "DHL", "6. Oktober 2026"],
            [],
        ),
        (
            "NE-100252",
            Intent.ORDER_STATUS,
            Behavior.ACTION,
            ["48 Stunden", "fehlgeschlagen"],
            ["faq-payment-failed"],
        ),
        (
            "NE-100245",
            Intent.ORDER_CANCEL,
            Behavior.ANSWER,
            ["kann kostenlos storniert", "Kundenkonto", ACCOUNT_URL, "54,94 €"],
            ["policy-order-cancellation"],
        ),
        (
            "NE-100250",
            Intent.ORDER_CANCEL,
            Behavior.ANSWER,
            ["bereits versendet", "Annahme"],
            ["policy-order-cancellation"],
        ),
        (
            "NE-100170",
            Intent.ORDER_CANCEL,
            Behavior.ANSWER,
            ["bereits storniert"],
            ["policy-order-cancellation"],
        ),
        (
            "NE-100231",
            Intent.RETURN_REQUEST,
            Behavior.ANSWER,
            ["20. Oktober 2026", "Retourenlabel", "Kundenkonto"],
            ["policy-returns-v5"],
        ),
        (
            "NE-100198",
            Intent.RETURN_REQUEST,
            Behavior.ANSWER,
            ["abgelaufen", "Gewährleistung"],
            ["policy-returns-v5", "policy-warranty"],
        ),
        (
            "NE-100240",
            Intent.RETURN_REQUEST,
            Behavior.ANSWER,
            ["bereits eine Rücksendung"],
            ["policy-returns-v5"],
        ),
        (
            "NE-100240",
            Intent.REFUND_STATUS,
            Behavior.ACTION,
            ["14 Tagen", "auf dem Weg"],
            ["faq-refund-timeline"],
        ),
        (
            "NE-100180",
            Intent.REFUND_STATUS,
            Behavior.ACTION,
            ["229,00 €", "28. September 2026"],
            ["faq-refund-timeline"],
        ),
    ],
)
def test_guidance_per_order(
    order_id: str, intent: Intent, behavior: Behavior, phrases: list[str], sources: list[str]
) -> None:
    response = order_guidance(intent, ORDERS[order_id], TODAY)

    assert response.behavior is behavior and response.sources == sources
    assert response.tool_calls == ["get_order_status"]
    for phrase in phrases:
        assert phrase in response.answer, phrase
    assert "wurde storniert" not in response.answer or order_id == "NE-100170"


def test_guidance_never_claims_to_have_acted() -> None:
    for order in ORDERS.values():
        for intent in (Intent.ORDER_CANCEL, Intent.RETURN_REQUEST):
            answer = order_guidance(intent, order, TODAY).answer
            assert "habe ich storniert" not in answer and "Retoure wurde erstellt" not in answer


def test_not_found_wording_reveals_nothing() -> None:
    response = order_not_found("NE-100250", Intent.ORDER_STATUS)

    assert response.behavior is Behavior.REFUSE
    assert "keine Bestellung NE-100250" in response.answer and "Velto" not in response.answer
