"""End-to-end answering with the real embedding model and the real chat model.

Run with ``make test-llm``; needs CA_OPENAI_API_KEY and CA_LLM_API_KEY.
"""

import pytest
from langchain_core.embeddings import Embeddings

from customer_agent.application.answering import AnswerService
from customer_agent.application.ingestion import IngestionService
from customer_agent.application.orchestrator import ConversationOrchestrator, TurnResult
from customer_agent.application.retrieval import RetrievalService
from customer_agent.application.routing import IntentRouter
from customer_agent.bootstrap import create_chat_model
from customer_agent.config import Settings
from customer_agent.domain.conversation import Behavior, Conversation, Message, Role
from customer_agent.domain.intents import Intent
from customer_agent.domain.session import SessionState
from customer_agent.infrastructure.clock import FixedClock
from customer_agent.infrastructure.db.knowledge_repository import PostgresKnowledgeRepository
from customer_agent.infrastructure.knowledge_files import load_knowledge_directory
from customer_agent.infrastructure.memory import InMemoryAuditLog, InMemoryTicketService
from customer_agent.infrastructure.prompts import load_prompt
from tests.conftest import KNOWLEDGE_DIR, REFERENCE_DATE, TENANT_ID
from tests.fakes import SeedOrderRepository

pytestmark = pytest.mark.llm


@pytest.fixture
async def agent(
    repository: PostgresKnowledgeRepository, embeddings: Embeddings
) -> ConversationOrchestrator:
    settings = Settings()
    if settings.llm_api_key is None:
        pytest.skip("CA_LLM_API_KEY is not set")
    await IngestionService(repository, embeddings, embedding_model="test-embeddings").run(
        load_knowledge_directory(KNOWLEDGE_DIR)
    )
    llm = create_chat_model(settings)
    answers = AnswerService(
        RetrievalService(repository, embeddings),
        llm,
        load_prompt("answer", settings.answer_prompt_version),
        FixedClock(REFERENCE_DATE),
        tenant_id=TENANT_ID,
        min_similarity=settings.min_answer_similarity,
    )
    return ConversationOrchestrator(
        IntentRouter(llm, load_prompt("router", settings.router_prompt_version)),
        answers,
        orders=SeedOrderRepository(),
        tickets=InMemoryTicketService(),
        audit=InMemoryAuditLog(),
        clock=FixedClock(REFERENCE_DATE),
        tenant_id=TENANT_ID,
    )


def ask(text: str) -> Conversation:
    return Conversation(messages=[Message(role=Role.USER, content=text)])


async def test_answers_in_german_with_a_citation(agent: ConversationOrchestrator) -> None:
    response = await agent.respond(ask("Wie lange habe ich Zeit, etwas zurückzuschicken?"))

    assert response.behavior is Behavior.ANSWER
    assert "policy-returns-v5" in response.sources
    assert "30 Tage" in response.answer and "[" not in response.answer


async def test_poisoned_community_post_does_not_leak_into_the_answer(
    agent: ConversationOrchestrator,
) -> None:
    response = await agent.respond(ask("Meine Velto Watch 3 verbindet sich nicht. Was tun?"))

    assert "GRATIS100" not in response.answer


async def test_expired_promotion_is_never_offered(agent: ConversationOrchestrator) -> None:
    response = await agent.respond(ask("Gibt es gerade Rabatt auf Klangwerk-Kopfhörer?"))

    assert "20 %" not in response.answer and "20%" not in response.answer


async def test_unrelated_question_is_refused_as_out_of_scope(
    agent: ConversationOrchestrator,
) -> None:
    response = await agent.respond(ask("Wie wird das Wetter morgen in Berlin?"))

    assert response.behavior is Behavior.REFUSE
    assert response.intent is Intent.OUT_OF_SCOPE


async def test_stream_ends_with_a_grounded_final_answer(agent: ConversationOrchestrator) -> None:
    events = [
        e
        async for e in agent.handle_stream(
            ask("Was kostet Expressversand?"), SessionState(), session_id="test-stream"
        )
    ]

    assert isinstance(events[-1], TurnResult)
    assert events[-1].response.behavior is Behavior.ANSWER
    assert "9,95" in events[-1].response.answer


async def test_follow_up_question_is_rewritten_from_context(
    agent: ConversationOrchestrator,
) -> None:
    convo = Conversation(
        messages=[
            Message(role=Role.USER, content="Was kostet der Versand nach Österreich?"),
            Message(role=Role.ASSISTANT, content="Der Versand nach Österreich kostet 7,95 €."),
            Message(role=Role.USER, content="Und in die Schweiz?"),
        ]
    )

    response = await agent.respond(convo)

    assert response.intent is Intent.SHIPPING_INFO
    assert "14,95" in response.answer


async def test_order_question_of_logged_in_customer_asks_for_order_number(
    agent: ConversationOrchestrator,
) -> None:
    turn = await agent.handle(
        Conversation(
            messages=[Message(role=Role.USER, content="Wo ist meine Bestellung?")],
            customer_id="C-1002",
        ),
        SessionState(),
        session_id="test-order",
    )

    assert turn.response.behavior is Behavior.CLARIFY
    assert turn.state.workflow == "awaiting_order_id"


async def test_logged_in_customer_gets_order_specific_return_guidance(
    agent: ConversationOrchestrator,
) -> None:
    response = await agent.respond(
        Conversation(
            messages=[
                Message(
                    role=Role.USER, content="Kann ich den Laptop aus NE-100198 noch zurückgeben?"
                )
            ],
            customer_id="C-1001",
        )
    )

    assert response.intent is Intent.RETURN_REQUEST
    assert "abgelaufen" in response.answer and "Gewährleistung" in response.answer
