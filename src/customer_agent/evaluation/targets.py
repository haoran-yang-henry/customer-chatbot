"""Agents that the evaluation CLI can run against."""

from customer_agent.application.ports import CustomerServiceAgent
from customer_agent.domain.conversation import AgentResponse, Behavior, Conversation


class PlaceholderAgent:
    """Stands in for the real agent until Day 3: always says it cannot answer."""

    async def respond(self, conversation: Conversation) -> AgentResponse:
        return AgentResponse(
            answer="Dazu kann ich Ihnen leider noch keine Auskunft geben.",
            behavior=Behavior.NO_ANSWER,
        )


TARGETS: dict[str, type[CustomerServiceAgent]] = {
    "placeholder": PlaceholderAgent,
}
