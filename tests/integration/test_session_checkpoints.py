"""Served sessions against a real database: the transcript and state are LangGraph checkpoints
(rebuild phase 3), and they hold the masked message only (invariant I3)."""

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from customer_agent.application.chat import ChatService
from customer_agent.application.orchestrator import ConversationOrchestrator
from customer_agent.application.turn_graph import CHECKPOINT_TYPES
from customer_agent.cli.retention import purge
from customer_agent.domain.session import SessionState, WorkflowState
from customer_agent.infrastructure.checkpointer import PostgresCheckpointer
from customer_agent.infrastructure.db.session_repository import PostgresSessionStore
from tests.fakes import FakeRetriever, ScriptedRouterLLM, document, make_orchestrator

IBAN = "DE89 3704 0044 0532 0130 00"
ROUTING = (
    '{"intent": "ORDER_STATUS", "confidence": 0.9, "about_specific_order": true, '
    '"standalone_query": "Status?"}'
)


@pytest.fixture
async def checkpointer(database_url: str) -> AsyncIterator[PostgresCheckpointer]:
    checkpointer = PostgresCheckpointer(database_url, CHECKPOINT_TYPES)
    await checkpointer.start()
    await checkpointer.wait_ready()  # the checkpoint tables exist
    yield checkpointer
    await checkpointer.aclose()


@pytest.fixture
def agent(checkpointer: PostgresCheckpointer) -> ConversationOrchestrator:
    retriever = FakeRetriever(documents=[document("policy-order-cancellation", 0.7)])
    return make_orchestrator(ScriptedRouterLLM(ROUTING), retriever, checkpointer=checkpointer.saver)


def chat_service(engine: AsyncEngine, agent: ConversationOrchestrator) -> ChatService:
    sessions = PostgresSessionStore(async_sessionmaker(engine, expire_on_commit=False))
    return ChatService(agent, sessions, tenant_id="nordelektro")


async def test_turns_continue_from_the_checkpoint(
    engine: AsyncEngine, agent: ConversationOrchestrator
) -> None:
    chat = chat_service(engine, agent)
    first = await chat.chat(None, "C-1002", "Wo ist meine Bestellung?")

    await chat.chat(first.session_id, "C-1002", "NE-100250")

    messages, state = await agent.transcript(first.session_id)
    assert len(messages) == 4 and messages[2].content == "NE-100250"
    assert state.slots == {"order_id": "NE-100250"} and state.workflow is WorkflowState.IDLE


async def test_checkpoints_hold_the_masked_message_never_the_raw_one(
    engine: AsyncEngine, agent: ConversationOrchestrator
) -> None:
    turn = await chat_service(engine, agent).chat(
        None, "C-1001", f"Wo ist meine Bestellung? Meine IBAN ist {IBAN}."
    )

    async with engine.connect() as connection:
        rows = [
            *await connection.execute(
                text("SELECT checkpoint::text, metadata::text FROM checkpoints")
            ),
            *await connection.execute(text("SELECT blob FROM checkpoint_blobs")),
            *await connection.execute(text("SELECT blob FROM checkpoint_writes")),
        ]
    stored = b"".join(
        value if isinstance(value, bytes) else str(value).encode()
        for row in rows
        for value in row
        if value is not None
    )

    messages, _ = await agent.transcript(turn.session_id)
    assert "[IBAN entfernt]" in messages[0].content
    assert b"[IBAN entfernt]" in stored  # the search looks where the transcript is
    assert IBAN.encode() not in stored and IBAN.replace(" ", "").encode() not in stored


async def test_retention_deletes_old_conversations_with_their_checkpoints(
    engine: AsyncEngine, agent: ConversationOrchestrator, checkpointer: PostgresCheckpointer
) -> None:
    chat = chat_service(engine, agent)
    old = await chat.chat(None, "C-1002", "Wo ist meine Bestellung?")
    recent = await chat.chat(None, "C-1002", "Wo ist meine Bestellung?")
    async with engine.begin() as connection:
        await connection.execute(
            text(
                "UPDATE chat_sessions SET updated_at = now() - interval '100 days' WHERE id = :id"
            ),
            {"id": old.session_id},
        )
    sessions = PostgresSessionStore(async_sessionmaker(engine, expire_on_commit=False))

    deleted = await purge(sessions, checkpointer, datetime.now(UTC) - timedelta(days=90))

    assert deleted == 1
    assert await sessions.load(old.session_id) is None
    assert await agent.transcript(old.session_id) == ([], SessionState())
    assert await sessions.load(recent.session_id) is not None
    messages, _ = await agent.transcript(recent.session_id)
    assert len(messages) == 2
