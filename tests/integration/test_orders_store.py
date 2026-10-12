"""Postgres adapters for orders, tickets and the audit log (migration 0004)."""

import json

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from customer_agent.application.ports import AuditEntry, NewTicket
from customer_agent.domain.orders import OrderStatus
from customer_agent.infrastructure.db.order_repository import (
    PostgresAuditLog,
    PostgresOrderRepository,
    PostgresTicketService,
    seed_orders,
)
from tests.conftest import SEED_DIR, TENANT_ID


@pytest.fixture
async def db(engine: AsyncEngine) -> async_sessionmaker:  # type: ignore[type-arg]
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    customers = json.loads((SEED_DIR / "customers.json").read_text("utf-8"))
    orders = json.loads((SEED_DIR / "orders.json").read_text("utf-8"))
    await seed_orders(sessions, customers, orders)
    await seed_orders(sessions, customers, orders)  # idempotent
    return sessions


async def test_seed_is_idempotent(db: async_sessionmaker, engine: AsyncEngine) -> None:  # type: ignore[type-arg]
    async with engine.connect() as connection:
        assert await connection.scalar(text("SELECT count(*) FROM orders")) == 9


async def test_own_order_is_returned_with_typed_fields(db: async_sessionmaker) -> None:  # type: ignore[type-arg]
    order = await PostgresOrderRepository(db).get_for_customer(
        "NE-100250", customer_id="C-1002", tenant_id=TENANT_ID
    )

    assert order is not None and order.status is OrderStatus.SHIPPED
    assert str(order.total) == "599.00" and order.shipping.expected_delivery is not None


@pytest.mark.parametrize(("customer", "tenant"), [("C-1001", TENANT_ID), ("C-1002", "other-shop")])
async def test_foreign_customer_or_tenant_gets_nothing(
    db: async_sessionmaker,  # type: ignore[type-arg]
    customer: str,
    tenant: str,
) -> None:
    order = await PostgresOrderRepository(db).get_for_customer(
        "NE-100250", customer_id=customer, tenant_id=tenant
    )

    assert order is None


async def test_tickets_get_sequential_public_numbers(db: async_sessionmaker) -> None:  # type: ignore[type-arg]
    tickets = PostgresTicketService(db)
    ticket = NewTicket(
        tenant_id=TENANT_ID,
        customer_id="C-1001",
        session_id="s",
        intent="HUMAN_AGENT",
        order_id=None,
        summary="x",
    )

    first, second = await tickets.create(ticket), await tickets.create(ticket)

    assert first.startswith("T-") and int(second[2:]) == int(first[2:]) + 1


async def test_audit_entries_are_appended(db: async_sessionmaker, engine: AsyncEngine) -> None:  # type: ignore[type-arg]
    entry = AuditEntry(
        tool="get_order_status",
        customer_id="C-1001",
        session_id="s",
        request_id="r",
        arguments={"order_id": "NE-100250"},
        outcome="not_found",
    )

    await PostgresAuditLog(db).record(entry)

    async with engine.connect() as connection:
        row = (
            await connection.execute(text("SELECT tool, arguments, outcome FROM audit_log"))
        ).one()
    assert row == ("get_order_status", {"order_id": "NE-100250"}, "not_found")
