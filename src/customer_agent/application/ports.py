"""Ports (interfaces) implemented by infrastructure adapters. Added as use cases need them."""

from typing import Protocol

from customer_agent.domain.conversation import AgentResponse, Conversation


class CustomerServiceAgent(Protocol):
    """Anything that can answer the next turn of a conversation: the real agent or a test double."""

    async def respond(self, conversation: Conversation) -> AgentResponse: ...
