"""PostgreSQL adapters for orders (read-only), tickets and the audit log."""

from collections.abc import Sequence
from datetime import date, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from customer_agent.application.ports import AuditEntry, NewTicket
from customer_agent.domain.orders import Order
from customer_agent.infrastructure.db.models import AuditLogRow, CustomerRow, OrderRow, TicketRow


def _order(row: OrderRow) -> Order:
    return Order.model_validate(
        {
            "order_id": row.order_id,
            "customer_id": row.customer_id,
            "status": row.status,
            "created_at": row.created_at,
            "items": row.items,
            "total": row.total,
            "payment_method": row.payment_method,
            "payment_status": row.payment_status,
            "shipping": row.shipping,
            "return_info": row.return_info,
        }
    )


class PostgresOrderRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def get_for_customer(
        self, order_id: str, *, customer_id: str, tenant_id: str
    ) -> Order | None:
        # Ownership is part of the query: a foreign order is never loaded at all.
        stmt = select(OrderRow).where(
            OrderRow.order_id == order_id,
            OrderRow.customer_id == customer_id,
            OrderRow.tenant_id == tenant_id,
        )
        async with self._sessions() as session:
            row = await session.scalar(stmt)
            return _order(row) if row else None


class PostgresTicketService:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def create(self, ticket: NewTicket) -> str:
        row = TicketRow(**ticket.model_dump())
        async with self._sessions.begin() as session:
            session.add(row)
            await session.flush()  # assigns the identity value
            return f"T-{row.id:06d}"

    async def recent(self, *, customer_id: str, tenant_id: str) -> list[str]:
        # The database clock decides "the last 24 hours", like it stamps created_at.
        stmt = (
            select(TicketRow.id)
            .where(
                TicketRow.customer_id == customer_id,
                TicketRow.tenant_id == tenant_id,
                TicketRow.created_at > func.now() - timedelta(hours=24),
            )
            .order_by(TicketRow.id.desc())
        )
        async with self._sessions() as session:
            return [f"T-{ticket_id:06d}" for ticket_id in await session.scalars(stmt)]


class PostgresAuditLog:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def record(self, entry: AuditEntry) -> None:
        async with self._sessions.begin() as session:
            session.add(AuditLogRow(**entry.model_dump()))


async def seed_orders(
    sessions: async_sessionmaker[AsyncSession],
    customers: Sequence[dict[str, Any]],
    orders: Sequence[dict[str, Any]],
) -> tuple[int, int]:
    """Idempotent upsert of the mock order system (``make seed``)."""
    async with sessions.begin() as session:
        for customer in customers:
            values = {
                k: customer[k]
                for k in ("customer_id", "tenant_id", "first_name", "last_name", "email", "country")
            }
            stmt = insert(CustomerRow).values(**values)
            await session.execute(
                stmt.on_conflict_do_update(index_elements=["customer_id"], set_=values)
            )
        tenant_by_customer = {c["customer_id"]: c["tenant_id"] for c in customers}
        for order in orders:
            Order.model_validate(order)  # reject seed data the domain could not read
            values = {
                "order_id": order["order_id"],
                "customer_id": order["customer_id"],
                "tenant_id": tenant_by_customer[order["customer_id"]],
                "status": order["status"],
                "created_at": date.fromisoformat(order["created_at"]),
                "items": order["items"],
                "total": Decimal(order["total"]),
                "payment_method": order["payment_method"],
                "payment_status": order["payment_status"],
                "shipping": order["shipping"],
                "return_info": order["return_info"],
            }
            stmt = insert(OrderRow).values(**values)
            await session.execute(
                stmt.on_conflict_do_update(index_elements=["order_id"], set_=values)
            )
    return len(customers), len(orders)
