"""Eligibility rules: every status, both window boundaries (FR-23, NFR-06)."""

from datetime import date
from decimal import Decimal
from typing import Any

import pytest

from customer_agent.domain.orders import (
    RETURN_WINDOW_DAYS,
    Order,
    OrderStatus,
    can_cancel,
    can_change_address,
    can_return,
)


def order(status: OrderStatus, delivered: date | None = None, sku: str = "VP-20") -> Order:
    data: dict[str, Any] = {
        "order_id": "NE-100001",
        "customer_id": "C-1",
        "status": status,
        "created_at": date(2026, 9, 1),
        "items": [{"sku": sku, "name": "X", "quantity": 1, "unit_price": "10.00"}],
        "total": Decimal("10.00"),
        "payment_method": "paypal",
        "payment_status": "captured",
        "shipping": {"carrier": "DHL", "delivered_at": delivered},
    }
    return Order.model_validate(data)


@pytest.mark.parametrize(
    ("status", "allowed", "reason"),
    [
        (OrderStatus.PENDING_PAYMENT, True, "not_shipped"),
        (OrderStatus.PAID, True, "not_shipped"),
        (OrderStatus.PROCESSING, True, "not_shipped"),
        (OrderStatus.SHIPPED, False, "already_shipped"),
        (OrderStatus.DELIVERED, False, "already_shipped"),
        (OrderStatus.RETURN_REQUESTED, False, "already_shipped"),
        (OrderStatus.RETURNED, False, "already_shipped"),
        (OrderStatus.CANCELLED, False, "already_cancelled"),
    ],
)
def test_can_cancel_covers_every_status(status: OrderStatus, allowed: bool, reason: str) -> None:
    eligibility = can_cancel(order(status))

    assert (eligibility.allowed, eligibility.reason) == (allowed, reason)


@pytest.mark.parametrize(
    ("status", "allowed", "reason"),
    [
        (OrderStatus.PENDING_PAYMENT, True, "not_processed"),
        (OrderStatus.PAID, True, "not_processed"),
        (OrderStatus.PROCESSING, False, "processing"),
        (OrderStatus.SHIPPED, False, "shipped"),
        (OrderStatus.DELIVERED, False, "delivered"),
        (OrderStatus.RETURN_REQUESTED, False, "delivered"),
        (OrderStatus.RETURNED, False, "delivered"),
        (OrderStatus.CANCELLED, False, "cancelled"),
    ],
)
def test_can_change_address_covers_every_status(
    status: OrderStatus, allowed: bool, reason: str
) -> None:
    # faq-order-change: only while "Zahlung ausstehend" or "Bezahlt".
    eligibility = can_change_address(order(status))

    assert (eligibility.allowed, eligibility.reason) == (allowed, reason)


DELIVERED = date(2026, 9, 20)


@pytest.mark.parametrize(
    ("today", "allowed"),
    [
        (date(2026, 9, 20), True),  # delivery day
        (date(2026, 10, 20), True),  # day 30: last day (inclusive)
        (date(2026, 10, 21), False),  # day 31
    ],
)
def test_return_window_boundaries(today: date, allowed: bool) -> None:
    eligibility = can_return(order(OrderStatus.DELIVERED, DELIVERED), today)

    assert eligibility.allowed is allowed
    assert eligibility.deadline == date(2026, 10, 20)
    assert (eligibility.deadline - DELIVERED).days == RETURN_WINDOW_DAYS


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        (OrderStatus.PENDING_PAYMENT, "not_delivered"),
        (OrderStatus.PAID, "not_delivered"),
        (OrderStatus.PROCESSING, "not_delivered"),
        (OrderStatus.SHIPPED, "not_delivered"),
        (OrderStatus.RETURN_REQUESTED, "return_in_progress"),
        (OrderStatus.RETURNED, "already_returned"),
        (OrderStatus.CANCELLED, "cancelled"),
    ],
)
def test_return_is_not_possible_outside_delivered(status: OrderStatus, reason: str) -> None:
    eligibility = can_return(order(status), date(2026, 10, 5))

    assert (eligibility.allowed, eligibility.reason) == (False, reason)


def test_delivered_without_date_never_guesses_in_favour_of_return() -> None:
    eligibility = can_return(order(OrderStatus.DELIVERED, None), date(2026, 10, 5))

    assert (eligibility.allowed, eligibility.reason) == (False, "delivery_date_unknown")


def test_in_ear_headphones_need_the_hygiene_seal() -> None:
    assert order(OrderStatus.DELIVERED, DELIVERED, sku="KW-BUDSS").needs_hygiene_seal
    assert not order(OrderStatus.DELIVERED, DELIVERED, sku="KW-ANC700").needs_hygiene_seal
