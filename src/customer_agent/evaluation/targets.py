"""Agents the evaluation CLI can run against, as async context managers (they own resources)."""

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from customer_agent.application.ports import CustomerServiceAgent
from customer_agent.domain.conversation import AgentResponse, Behavior, Conversation
from customer_agent.evaluation.schema import DATASET_REFERENCE_DATE

Target = tuple[CustomerServiceAgent, dict[str, str]]


class PlaceholderAgent:
    """Stands in for the real agent: always says it cannot answer."""

    async def respond(self, conversation: Conversation) -> AgentResponse:
        return AgentResponse(
            answer="Dazu kann ich Ihnen leider noch keine Auskunft geben.",
            behavior=Behavior.NO_ANSWER,
        )


@asynccontextmanager
async def _placeholder() -> AsyncIterator[Target]:
    yield PlaceholderAgent(), {}


@asynccontextmanager
async def _agent() -> AsyncIterator[Target]:
    """The real agent with a fixed clock: dataset expectations assume the reference date."""
    from customer_agent.bootstrap import build_container
    from customer_agent.config import get_settings
    from customer_agent.infrastructure.clock import FixedClock

    settings = get_settings()
    container = build_container(settings, clock=FixedClock(DATASET_REFERENCE_DATE))
    try:
        config = {
            "llm_model": container.llm_model,
            "answer_prompt": container.answer_prompt,
            "embedding_model": settings.embedding_model,
            "min_answer_similarity": str(settings.min_answer_similarity),
        }
        yield container.agent, config
    finally:
        await container.aclose()


TARGETS: dict[str, Callable[[], AbstractAsyncContextManager[Target]]] = {
    "placeholder": _placeholder,
    "agent": _agent,
}
