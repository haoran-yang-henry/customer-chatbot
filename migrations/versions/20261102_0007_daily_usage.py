"""Daily usage: chat messages per UTC day from all clients together.

The hosted demo is public; ``CA_DAILY_MESSAGE_LIMIT`` caps what a day may cost. One row per day,
counted up in the same statement that reads it, so concurrent requests cannot both slip through.

Revision ID: 0007
Revises: 0006
Create Date: 2026-11-02
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "daily_usage",
        sa.Column("day", sa.Date(), primary_key=True),
        sa.Column("messages", sa.Integer(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("daily_usage")
