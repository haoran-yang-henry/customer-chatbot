from datetime import date

import pytest

from customer_agent.application.answering import AnswerService
from customer_agent.application.orchestrator import ConversationOrchestrator, TurnResult
from customer_agent.application.routing import IntentRouter
from customer_agent.domain.conversation import Behavior, Conversation, Message, Role
from customer_agent.domain.intents import Intent
from customer_agent.domain.session import SessionState, WorkflowState
from customer_agent.infrastructure.prompts import load_prompt
from tests.fakes import FakeRetriever, FixedClock, ScriptedRouterLLM, document

RETRIEVER = FakeRetriever([document("policy-returns-v5", 0.7)])


def orchestrator(llm: ScriptedRouterLLM) -> ConversationOrchestrator:
    RETRIEVER.queries.clear()
    answers = AnswerService(
        RETRIEVER,
        llm,
        load_prompt("answer", 2),
        FixedClock(date(2026, 10, 5)),
        tenant_id="nordelektro",
        min_similarity=0.35,
    )
    return ConversationOrchestrator(IntentRouter(llm, load_prompt("router", 1)), answers)


def ask(text: str, customer: str | None = None) -> Conversation:
    return Conversation(messages=[Message(role=Role.USER, content=text)], customer_id=customer)


def routing(intent: str, specific: bool = False) -> str:
    return (
        f'{{"intent": "{intent}", "confidence": 0.9, "about_specific_order": '
        f'{"true" if specific else "false"}, "standalone_query": "Rückgabe Kosten?"}}'
    )


async def test_knowledge_intent_is_answered_with_rewritten_query_and_intent() -> None:
    llm = ScriptedRouterLLM(routing("RETURN_POLICY"))

    response = await orchestrator(llm).respond(ask("Und was kostet das?"))

    assert response.behavior is Behavior.ANSWER and response.intent is Intent.RETURN_POLICY
    assert RETRIEVER.queries[0][0] == "Rückgabe Kosten?"  # retrieval uses the rewrite
    assert "Und was kostet das?" in llm.calls[-1][1].content  # the answer sees the original


@pytest.mark.parametrize(
    ("intent", "behavior"),
    [
        ("OUT_OF_SCOPE", Behavior.REFUSE),
        ("HUMAN_AGENT", Behavior.ESCALATE),
        ("COMPLAINT", Behavior.ESCALATE),
    ],
)
async def test_template_steps_do_not_call_the_answer_llm(intent: str, behavior: Behavior) -> None:
    llm = ScriptedRouterLLM(routing(intent))

    response = await orchestrator(llm).respond(ask("…"))

    assert response.behavior is behavior and response.intent == intent
    assert len(llm.calls) == 0  # only the router ran (router calls are not recorded in calls)


async def test_greeting_needs_no_llm_at_all() -> None:
    llm = ScriptedRouterLLM("unused")

    response = await orchestrator(llm).respond(ask("Hallo!"))

    assert response.intent is Intent.GREETING and "digitale Assistent" in response.answer
    assert llm.router_calls == 0


async def test_order_question_waits_for_order_number_then_continues() -> None:
    llm = ScriptedRouterLLM(routing("ORDER_STATUS", specific=True))
    agent = orchestrator(llm)

    first = await agent.handle(
        ask("Wo ist meine Bestellung?", "C-1"), SessionState(), session_id="s"
    )
    assert first.response.behavior is Behavior.CLARIFY
    assert first.state.workflow is WorkflowState.AWAITING_ORDER_ID

    convo = Conversation(
        messages=[
            Message(role=Role.USER, content="Wo ist meine Bestellung?"),
            Message(role=Role.ASSISTANT, content=first.response.answer),
            Message(role=Role.USER, content="NE-100250"),
        ],
        customer_id="C-1",
    )
    second = await agent.handle(convo, first.state, session_id="s")

    assert llm.router_calls == 1  # the bare order number was routed by rules
    assert second.state.slots == {"order_id": "NE-100250"}
    assert second.response.intent is Intent.ORDER_STATUS


async def test_anonymous_order_question_asks_to_log_in() -> None:
    response = await orchestrator(ScriptedRouterLLM(routing("ORDER_STATUS", True))).respond(
        ask("Wo ist NE-100250?")
    )

    assert response.behavior is Behavior.REFUSE and "angemeldet" in response.answer


async def test_oversized_input_is_blocked_before_routing() -> None:
    llm = ScriptedRouterLLM("unused")

    response = await orchestrator(llm).respond(ask("x" * 5000))

    assert response.reasons == ["input_too_long"] and llm.router_calls == 0


async def test_provider_outage_degrades_to_fallback() -> None:
    llm = ScriptedRouterLLM(routing("RETURN_POLICY"))
    llm.fail = True

    response = await orchestrator(llm).respond(ask("Kosten?"))

    assert response.reasons == ["provider_unavailable"]


async def test_stream_ends_with_turn_result() -> None:
    llm = ScriptedRouterLLM(routing("RETURN_POLICY"))

    events = [
        e
        async for e in orchestrator(llm).handle_stream(
            ask("Kosten?"), SessionState(), session_id="s"
        )
    ]

    assert isinstance(events[-1], TurnResult)
    assert events[-1].state.active_intent is Intent.RETURN_POLICY
