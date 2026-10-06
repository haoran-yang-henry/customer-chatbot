"""End-to-end answering with the real embedding model and the real chat model.

Run with ``make test-llm``; needs CA_OPENAI_API_KEY and CA_LLM_API_KEY.
"""

import pytest

from customer_agent.application.answering import AnswerService, FinalAnswer
from customer_agent.application.ingestion import IngestionService
from customer_agent.application.orchestrator import ConversationOrchestrator
from customer_agent.application.ports import EmbeddingProvider
from customer_agent.application.retrieval import RetrievalService
from customer_agent.bootstrap import create_llm_provider
from customer_agent.config import Settings
from customer_agent.domain.conversation import Behavior, Conversation, Message, Role
from customer_agent.infrastructure.clock import FixedClock
from customer_agent.infrastructure.db.knowledge_repository import PostgresKnowledgeRepository
from customer_agent.infrastructure.knowledge_files import load_knowledge_directory
from customer_agent.infrastructure.prompts import load_prompt
from tests.conftest import KNOWLEDGE_DIR, REFERENCE_DATE, TENANT_ID

pytestmark = pytest.mark.llm


@pytest.fixture
async def agent(
    repository: PostgresKnowledgeRepository, embeddings: EmbeddingProvider
) -> ConversationOrchestrator:
    settings = Settings()
    if settings.llm_api_key is None:
        pytest.skip("CA_LLM_API_KEY is not set")
    await IngestionService(repository, embeddings).run(load_knowledge_directory(KNOWLEDGE_DIR))
    answers = AnswerService(
        RetrievalService(repository, embeddings),
        create_llm_provider(settings),
        load_prompt("answer", settings.answer_prompt_version),
        FixedClock(REFERENCE_DATE),
        tenant_id=TENANT_ID,
        min_similarity=settings.min_answer_similarity,
    )
    return ConversationOrchestrator(answers)


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


async def test_unrelated_question_gets_no_answer(agent: ConversationOrchestrator) -> None:
    response = await agent.respond(ask("Wie wird das Wetter morgen in Berlin?"))

    assert response.behavior is Behavior.NO_ANSWER


async def test_stream_ends_with_a_grounded_final_answer(agent: ConversationOrchestrator) -> None:
    events = [e async for e in agent.respond_stream(ask("Was kostet Expressversand?"))]

    assert isinstance(events[-1], FinalAnswer)
    assert events[-1].response.behavior is Behavior.ANSWER
    assert "9,95" in events[-1].response.answer
