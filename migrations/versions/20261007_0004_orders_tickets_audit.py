"""Mock order system (read-only for the agent), handoff tickets and the audit log.

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-07
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "customers",
        sa.Column("customer_id", sa.Text(), primary_key=True),
        sa.Column("tenant_id", sa.Text(), nullable=False),
        sa.Column("first_name", sa.Text(), nullable=False),
        sa.Column("last_name", sa.Text(), nullable=False),
        sa.Column("email", sa.Text(), nullable=False),
        sa.Column("country", sa.Text(), nullable=False),
    )
    op.create_table(
        "orders",
        sa.Column("order_id", sa.Text(), primary_key=True),
        sa.Column("customer_id", sa.Text(), sa.ForeignKey("customers.customer_id"), nullable=False),
        sa.Column("tenant_id", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Date(), nullable=False),
        sa.Column("items", JSONB(), nullable=False),
        sa.Column("total", sa.Numeric(10, 2), nullable=False),
        sa.Column("payment_method", sa.Text(), nullable=False),
        sa.Column("payment_status", sa.Text(), nullable=False),
        sa.Column("shipping", JSONB(), nullable=False),
        sa.Column("return_info", JSONB(), nullable=True),
    )
    # Every agent lookup filters by (order_id, customer_id, tenant_id).
    op.create_index("ix_orders_customer_id", "orders", ["customer_id"])
    op.create_table(
        "tickets",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("tenant_id", sa.Text(), nullable=False),
        sa.Column("customer_id", sa.Text(), nullable=True),
        # No FK: evaluation sessions are not persisted.
        sa.Column("session_id", sa.Text(), nullable=False),
        sa.Column("intent", sa.Text(), nullable=True),
        sa.Column("order_id", sa.Text(), nullable=True),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default="open"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_table(
        "audit_log",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("request_id", sa.Text(), nullable=False),
        sa.Column("session_id", sa.Text(), nullable=False),
        sa.Column("customer_id", sa.Text(), nullable=True),
        sa.Column("tool", sa.Text(), nullable=False),
        sa.Column("arguments", JSONB(), nullable=False),
        sa.Column("outcome", sa.Text(), nullable=False),
    )
    # Append-only: the application role never needs UPDATE/DELETE on the audit log (T14).
    op.create_index("ix_audit_log_customer_at", "audit_log", ["customer_id", "at"])


def downgrade() -> None:
    op.drop_table("audit_log")
    op.drop_table("tickets")
    op.drop_table("orders")
    op.drop_table("customers")
