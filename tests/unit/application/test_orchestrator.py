from datetime import date

from customer_agent.application.answering import AnswerService, FinalAnswer
from customer_agent.application.orchestrator import ConversationOrchestrator
from customer_agent.domain.conversation import Behavior, Conversation, Message, Role
from customer_agent.infrastructure.prompts import load_prompt
from tests.fakes import FakeLLM, FakeRetriever, FixedClock, document


def orchestrator(llm: FakeLLM) -> ConversationOrchestrator:
    answers = AnswerService(
        FakeRetriever([document("policy-returns-v5", 0.7)]),
        llm,
        load_prompt("answer", 1),
        FixedClock(date(2026, 10, 5)),
        tenant_id="nordelektro",
        min_similarity=0.5,
    )
    return ConversationOrchestrator(answers)


def ask(text: str) -> Conversation:
    return Conversation(messages=[Message(role=Role.USER, content=text)])


async def test_oversized_input_is_blocked_before_retrieval_and_llm() -> None:
    llm = FakeLLM("unused")

    response = await orchestrator(llm).respond(ask("x" * 5000))

    assert response.behavior is Behavior.CLARIFY
    assert response.reasons == ["input_too_long"]
    assert llm.calls == []


async def test_provider_outage_degrades_to_fallback_instead_of_error() -> None:
    response = await orchestrator(FakeLLM(fail=True)).respond(ask("Kosten Rücksendung?"))

    assert response.behavior is Behavior.NO_ANSWER
    assert response.reasons == ["provider_unavailable"]


async def test_provider_outage_while_streaming_ends_with_fallback() -> None:
    events = [e async for e in orchestrator(FakeLLM(fail=True)).respond_stream(ask("Kosten?"))]

    assert len(events) == 1 and isinstance(events[0], FinalAnswer)
    assert events[0].response.reasons == ["provider_unavailable"]
