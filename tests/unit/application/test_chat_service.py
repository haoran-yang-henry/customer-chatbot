from datetime import date

import pytest

from customer_agent.application.answering import AnswerService
from customer_agent.application.chat import ChatService, SessionNotFoundError
from customer_agent.application.orchestrator import ConversationOrchestrator
from customer_agent.application.routing import IntentRouter
from customer_agent.domain.session import WorkflowState
from customer_agent.infrastructure.prompts import load_prompt
from tests.fakes import FakeRetriever, FixedClock, InMemorySessionStore, ScriptedRouterLLM, document

ROUTING = (
    '{"intent": "ORDER_STATUS", "confidence": 0.9, "about_specific_order": true, '
    '"standalone_query": "Status?"}'
)


def service(store: InMemorySessionStore) -> ChatService:
    llm = ScriptedRouterLLM(ROUTING)
    answers = AnswerService(
        FakeRetriever([document("policy-order-cancellation", 0.7)]),
        llm,
        load_prompt("answer", 2),
        FixedClock(date(2026, 10, 5)),
        tenant_id="nordelektro",
        min_similarity=0.35,
    )
    agent = ConversationOrchestrator(IntentRouter(llm, load_prompt("router", 1)), answers)
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
    first = await chat.chat(None, "C-1001", "Wo ist meine Bestellung?")

    await chat.chat(first.session_id, "C-1001", "NE-100250")

    state = store.sessions[first.session_id].state
    assert (
        state.slots == {"order_id": "NE-100250"}
        and len(store.sessions[first.session_id].messages) == 4
    )


@pytest.mark.parametrize("session_id", ["does-not-exist", "s1"])
async def test_unknown_or_foreign_session_is_not_found(session_id: str) -> None:
    store = InMemorySessionStore()
    chat = service(store)
    await chat.chat(None, "C-1001", "Hallo")  # creates s1 for C-1001

    with pytest.raises(SessionNotFoundError):
        await chat.chat(session_id, "C-2002", "Zeig mir den Verlauf")
