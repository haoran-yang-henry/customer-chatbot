import asyncio

import pytest

from customer_agent.application import chat as chat_module
from customer_agent.application.chat import ChatService, SessionNotFoundError, TurnNotFoundError
from customer_agent.application.orchestrator import ConversationOrchestrator
from customer_agent.application.ports import ConcurrentTurnError
from customer_agent.domain.conversation import Role
from customer_agent.domain.session import SessionState, WorkflowState
from customer_agent.observability import request_id_var
from tests.fakes import (
    FakeRetriever,
    InMemorySessionStore,
    ScriptedRouterLLM,
    document,
    make_orchestrator,
)

ROUTING = (
    '{"intent": "ORDER_STATUS", "confidence": 0.9, "about_specific_order": true, '
    '"standalone_query": "Status?"}'
)


def service(store: InMemorySessionStore) -> tuple[ChatService, ConversationOrchestrator]:
    llm = ScriptedRouterLLM(ROUTING)
    retriever = FakeRetriever(documents=[document("policy-order-cancellation", 0.7)])
    agent = make_orchestrator(llm, retriever)
    return ChatService(agent, store, tenant_id="nordelektro"), agent


async def test_new_session_is_created_and_turn_checkpointed() -> None:
    store = InMemorySessionStore()
    chat, agent = service(store)

    turn = await chat.chat(None, "C-1001", "Wo ist meine Bestellung?")

    messages, state = await agent.transcript(turn.session_id)
    assert [m.content for m in messages] == ["Wo ist meine Bestellung?", turn.response.answer]
    assert [m.role for m in messages] == [Role.USER, Role.ASSISTANT]
    assert state.workflow is WorkflowState.AWAITING_ORDER_ID
    assert store.leased == set()  # released after the turn


async def test_second_turn_continues_from_the_checkpointed_state() -> None:
    store = InMemorySessionStore()
    chat, agent = service(store)
    first = await chat.chat(None, "C-1002", "Wo ist meine Bestellung?")

    await chat.chat(first.session_id, "C-1002", "NE-100250")

    messages, state = await agent.transcript(first.session_id)
    assert state.slots == {"order_id": "NE-100250"} and len(messages) == 4


async def test_an_order_that_was_not_found_is_not_remembered() -> None:
    # NE-100250 belongs to C-1002: it must not reach a later ticket of C-1001.
    store = InMemorySessionStore()
    chat, agent = service(store)
    first = await chat.chat(None, "C-1001", "Wo ist meine Bestellung?")

    await chat.chat(first.session_id, "C-1001", "NE-100250")

    _, state = await agent.transcript(first.session_id)
    assert "order_id" not in state.slots


@pytest.mark.parametrize("session_id", ["does-not-exist", "s1"])
async def test_unknown_or_foreign_session_is_not_found(session_id: str) -> None:
    store = InMemorySessionStore()
    chat, _ = service(store)
    await chat.chat(None, "C-1001", "Hallo")  # creates s1 for C-1001

    with pytest.raises(SessionNotFoundError):
        await chat.chat(session_id, "C-2002", "Zeig mir den Verlauf")


async def test_turn_is_checkpointed_even_if_the_client_disconnects_mid_stream() -> None:
    store = InMemorySessionStore()
    chat, agent = service(store)
    first = await chat.chat(None, "C-1001", "Wo ist meine Bestellung?")  # asks for order no.

    stream = chat.chat_stream(first.session_id, "C-1001", "Wie lange habe ich Zeit?")
    await anext(stream)  # the client reads one event …
    await stream.aclose()  # … and disconnects
    await chat.drain()

    messages, _ = await agent.transcript(first.session_id)
    assert len(messages) == 4  # the second turn was checkpointed
    assert store.leased == set()


async def test_a_turn_while_another_runs_is_rejected_before_it_starts() -> None:
    store = InMemorySessionStore()
    chat, agent = service(store)
    created = await store.create(tenant_id="nordelektro", customer_id="C-1001")
    await store.acquire(created.session_id)  # another turn is running

    with pytest.raises(ConcurrentTurnError):
        await chat.chat(created.session_id, "C-1001", "Wo ist meine Bestellung?")

    assert await agent.transcript(created.session_id) == ([], SessionState())
    assert store.leased == {created.session_id}  # the running turn keeps its lease


async def test_concurrent_turns_one_runs_the_other_is_rejected() -> None:
    store = InMemorySessionStore()
    chat, agent = service(store)
    created = await store.create(tenant_id="nordelektro", customer_id="C-1001")

    outcomes = await asyncio.gather(
        *(chat.chat(created.session_id, "C-1001", f"Frage {i}?") for i in range(3)),
        return_exceptions=True,
    )

    assert sum(not isinstance(o, BaseException) for o in outcomes) == 1
    assert all(isinstance(o, ConcurrentTurnError) for o in outcomes if isinstance(o, Exception))
    messages, _ = await agent.transcript(created.session_id)
    assert len(messages) == 2


async def test_session_of_another_tenant_is_not_found() -> None:
    store = InMemorySessionStore()
    foreign = await store.create(tenant_id="other-shop", customer_id="C-1001")

    with pytest.raises(SessionNotFoundError):
        await service(store)[0].chat(foreign.session_id, "C-1001", "Hallo")


async def test_transcript_stores_the_masked_message_never_the_raw_one() -> None:
    store = InMemorySessionStore()
    chat, agent = service(store)

    turn = await chat.chat(None, "C-1001", "Wo ist meine Bestellung? Rückruf unter 0171 2345678")

    messages, _ = await agent.transcript(turn.session_id)
    assert messages[0].content == "Wo ist meine Bestellung? Rückruf unter [Telefonnummer entfernt]"


# --- feedback --------------------------------------------------------------------------------

TRACE = "0af7651916cd43dd8448eb211c80319c"


@pytest.fixture
def scores(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, bool, str | None]]:
    """Langfuse scores sent; the turn runs in a trace with a fixed id, over HTTP (request id)."""
    sent: list[tuple[str, bool, str | None]] = []

    def score_turn(trace_id: str, *, helpful: bool, comment: str | None) -> None:
        sent.append((trace_id, helpful, comment))

    monkeypatch.setattr(chat_module, "score_turn", score_turn)
    monkeypatch.setattr(chat_module, "current_trace_id", lambda: TRACE)
    return sent


async def served_turn(chat: ChatService, request_id: str, session_id: str | None = None) -> str:
    token = request_id_var.set(request_id)  # what the HTTP middleware does
    try:
        turn = await chat.chat(session_id, "C-1001", "Wo ist meine Bestellung?")
    finally:
        request_id_var.reset(token)
    return turn.session_id


async def test_an_answer_is_rated_with_a_masked_comment_on_its_trace(
    scores: list[tuple[str, bool, str | None]],
) -> None:
    store = InMemorySessionStore()
    chat, _ = service(store)
    session_id = await served_turn(chat, "req-1")

    await chat.feedback(
        session_id, "C-1001", "req-1", helpful=False, comment="Falsch. Ruf an: 0171 2345678"
    )

    masked = "Falsch. Ruf an: [Telefonnummer entfernt]"
    assert store.ratings == {(session_id, "req-1"): (False, masked)}
    assert scores == [(TRACE, False, masked)]


async def test_a_later_rating_replaces_the_earlier_one(
    scores: list[tuple[str, bool, str | None]],
) -> None:
    store = InMemorySessionStore()
    chat, _ = service(store)
    session_id = await served_turn(chat, "req-1")

    await chat.feedback(session_id, "C-1001", "req-1", helpful=False)
    await chat.feedback(session_id, "C-1001", "req-1", helpful=True)

    assert store.ratings[(session_id, "req-1")] == (True, None)


async def test_only_the_sessions_customer_rates_its_answers(
    scores: list[tuple[str, bool, str | None]],
) -> None:
    store = InMemorySessionStore()
    chat, _ = service(store)
    session_id = await served_turn(chat, "req-1")

    with pytest.raises(SessionNotFoundError):
        await chat.feedback(session_id, "C-2002", "req-1", helpful=False)
    with pytest.raises(TurnNotFoundError):
        await chat.feedback(session_id, "C-1001", "req-unknown", helpful=False)
    assert store.ratings == {} and scores == []


async def test_a_turn_without_an_http_request_is_not_recorded() -> None:
    store = InMemorySessionStore()
    chat, _ = service(store)

    await chat.chat(None, "C-1001", "Wo ist meine Bestellung?")

    assert store.turns == {}
