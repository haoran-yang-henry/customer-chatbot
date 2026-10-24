from typing import Any

import pytest

from customer_agent.application.chat import ChatService, SessionNotFoundError
from customer_agent.application.ports import ConcurrentTurnError
from customer_agent.domain.session import WorkflowState
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


def service(store: InMemorySessionStore) -> ChatService:
    llm = ScriptedRouterLLM(ROUTING)
    agent = make_orchestrator(
        llm, FakeRetriever(documents=[document("policy-order-cancellation", 0.7)])
    )
    return ChatService(agent, store, tenant_id="nordelektro")


async def test_new_session_is_created_and_turn_persisted() -> None:
    store = InMemorySessionStore()

    turn = await service(store).chat(None, "C-1001", "Wo ist meine Bestellung?")

    saved = store.sessions[turn.session_id]
    assert [m.content for m in saved.messages] == ["Wo ist meine Bestellung?", turn.response.answer]
    assert saved.state.workflow is WorkflowState.AWAITING_ORDER_ID


async def test_second_turn_continues_from_stored_state() -> None:
    store = InMemorySessionStore()
    chat = service(store)
    first = await chat.chat(None, "C-1002", "Wo ist meine Bestellung?")

    await chat.chat(first.session_id, "C-1002", "NE-100250")

    state = store.sessions[first.session_id].state
    assert (
        state.slots == {"order_id": "NE-100250"}
        and len(store.sessions[first.session_id].messages) == 4
    )


async def test_an_order_that_was_not_found_is_not_remembered() -> None:
    # NE-100250 belongs to C-1002: it must not reach a later ticket of C-1001.
    store = InMemorySessionStore()
    chat = service(store)
    first = await chat.chat(None, "C-1001", "Wo ist meine Bestellung?")

    await chat.chat(first.session_id, "C-1001", "NE-100250")

    assert "order_id" not in store.sessions[first.session_id].state.slots


@pytest.mark.parametrize("session_id", ["does-not-exist", "s1"])
async def test_unknown_or_foreign_session_is_not_found(session_id: str) -> None:
    store = InMemorySessionStore()
    chat = service(store)
    await chat.chat(None, "C-1001", "Hallo")  # creates s1 for C-1001

    with pytest.raises(SessionNotFoundError):
        await chat.chat(session_id, "C-2002", "Zeig mir den Verlauf")


async def test_turn_is_saved_even_if_the_client_disconnects_mid_stream() -> None:
    store = InMemorySessionStore()
    chat = service(store)
    first = await chat.chat(None, "C-1001", "Wo ist meine Bestellung?")  # asks for order no.

    stream = chat.chat_stream(first.session_id, "C-1001", "Wie lange habe ich Zeit?")
    await anext(stream)  # the client reads one event …
    await stream.aclose()  # … and disconnects
    await chat.drain()

    assert len(store.sessions[first.session_id].messages) == 4  # second turn was persisted


class RacingStore(InMemorySessionStore):
    """Another turn commits between this turn's load and its save."""

    async def save_turn(self, session_id: str, **kwargs: Any) -> None:
        current = self.sessions[session_id]
        self.sessions[session_id] = current.model_copy(update={"version": current.version + 1})
        await super().save_turn(session_id, **kwargs)


async def test_concurrent_turn_on_the_same_session_is_rejected() -> None:
    store = RacingStore()
    chat = service(store)
    created = await store.create(tenant_id="nordelektro", customer_id="C-1001")

    with pytest.raises(ConcurrentTurnError):
        await chat.chat(created.session_id, "C-1001", "Wo ist meine Bestellung?")

    assert store.sessions[created.session_id].messages == []  # nothing half-written


async def test_session_of_another_tenant_is_not_found() -> None:
    store = InMemorySessionStore()
    foreign = await store.create(tenant_id="other-shop", customer_id="C-1001")

    with pytest.raises(SessionNotFoundError):
        await service(store).chat(foreign.session_id, "C-1001", "Hallo")


async def test_transcript_stores_the_masked_message_never_the_raw_one() -> None:
    store = InMemorySessionStore()

    turn = await service(store).chat(
        None, "C-1001", "Wo ist meine Bestellung? Rückruf unter 0171 2345678"
    )

    saved = store.sessions[turn.session_id].messages[0].content
    assert saved == "Wo ist meine Bestellung? Rückruf unter [Telefonnummer entfernt]"
