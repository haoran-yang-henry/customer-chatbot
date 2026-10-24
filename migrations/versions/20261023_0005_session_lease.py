"""Chat sessions keep ownership and a turn lease; transcript and state move to the checkpoints.

LangGraph's checkpointer (rebuild phase 3) stores the masked transcript and the session state
per thread (= session id) in its own tables, which it creates and versions itself on start-up.
The lease replaces the optimistic version: a turn takes it before it starts, so a concurrent
turn is rejected (409) before any model call, not after.

Existing transcripts and states are dropped, not migrated: a session continues with an empty
history.

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-23
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import ARRAY, JSONB

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "chat_sessions", sa.Column("turn_started_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.drop_table("chat_messages")
    for column in ("active_intent", "workflow_state", "slots", "version"):
        op.drop_column("chat_sessions", column)


def downgrade() -> None:
    op.add_column("chat_sessions", sa.Column("active_intent", sa.Text(), nullable=True))
    op.add_column(
        "chat_sessions",
        sa.Column("workflow_state", sa.Text(), nullable=False, server_default="idle"),
    )
    op.add_column(
        "chat_sessions",
        sa.Column("slots", JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
    )
    op.add_column(
        "chat_sessions",
        sa.Column("version", sa.Integer(), nullable=False, server_default=sa.text("0")),
    )
    op.create_table(
        "chat_messages",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column(
            "session_id",
            sa.Text(),
            sa.ForeignKey("chat_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("intent", sa.Text(), nullable=True),
        sa.Column("behavior", sa.Text(), nullable=True),
        sa.Column("sources", ARRAY(sa.Text()), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint("session_id", "position"),
    )
    op.create_index("ix_chat_messages_session_id", "chat_messages", ["session_id"])
    op.drop_column("chat_sessions", "turn_started_at")
