"""Agents the evaluation CLI can run against, as async context managers (they own resources)."""

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import TYPE_CHECKING

from customer_agent.application.ports import CustomerServiceAgent
from customer_agent.domain.conversation import AgentResponse, Behavior, Conversation
from customer_agent.evaluation.schema import DATASET_REFERENCE_DATE

if TYPE_CHECKING:
    from customer_agent.config import Settings

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


async def _require_seeded_orders(settings: "Settings") -> None:
    """Without seed data every order lookup is 'not found' and transactional cases fail
    for the wrong reason."""
    from sqlalchemy import text

    from customer_agent.bootstrap import ConfigurationError, create_engine

    engine = create_engine(settings)
    try:
        async with engine.connect() as connection:
            orders = await connection.scalar(text("SELECT count(*) FROM orders"))
    finally:
        await engine.dispose()
    if not orders:
        raise ConfigurationError("the orders table is empty: run `make seed` before `make eval`")


@asynccontextmanager
async def _agent() -> AsyncIterator[Target]:
    """The real agent with a fixed clock: dataset expectations assume the reference date."""
    from customer_agent.bootstrap import build_container
    from customer_agent.config import get_settings
    from customer_agent.infrastructure.clock import FixedClock

    settings = get_settings()
    await _require_seeded_orders(settings)
    container = build_container(
        settings, clock=FixedClock(DATASET_REFERENCE_DATE), side_effects=False
    )
    try:
        config = {
            **{f"{role}_model": model for role, model in container.models.items()},
            **{f"{name}_prompt": ref for name, ref in container.prompts.items()},
            "embedding_model": settings.embedding_model,
            "min_answer_similarity": str(settings.min_answer_similarity),
            "guard_classifier": (
                f"on (threshold {settings.guard_classifier_threshold})"
                if settings.guard_classifier_enabled
                else "off"
            ),
        }
        yield container.agent, config
    finally:
        await container.aclose()


TARGETS: dict[str, Callable[[], AbstractAsyncContextManager[Target]]] = {
    "placeholder": _placeholder,
    "agent": _agent,
}
