import json
from decimal import Decimal
from typing import Any

import pytest

from tests.conftest import SEED_DIR

ORDER_STATUSES = {
    "pending_payment",
    "paid",
    "processing",
    "shipped",
    "delivered",
    "cancelled",
    "return_requested",
    "returned",
}


@pytest.fixture(scope="module")
def orders() -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = json.loads((SEED_DIR / "orders.json").read_text("utf-8"))
    return data


def test_orders_reference_existing_customers(orders: list[dict[str, Any]]) -> None:
    customers = json.loads((SEED_DIR / "customers.json").read_text("utf-8"))
    known = {customer["customer_id"] for customer in customers}

    assert {order["customer_id"] for order in orders} <= known


def test_order_ids_are_unique(orders: list[dict[str, Any]]) -> None:
    ids = [order["order_id"] for order in orders]

    assert len(ids) == len(set(ids))


def test_order_statuses_are_known(orders: list[dict[str, Any]]) -> None:
    assert {order["status"] for order in orders} <= ORDER_STATUSES


def test_totals_cover_item_prices(orders: list[dict[str, Any]]) -> None:
    """Total = items + shipping, so it can never be below the item sum."""
    for order in orders:
        items = sum(Decimal(i["unit_price"]) * i["quantity"] for i in order["items"])
        assert Decimal(order["total"]) >= items, order["order_id"]
