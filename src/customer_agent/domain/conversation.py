"""Conversation and agent response contracts shared by the API, the agent and the evaluation."""

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from customer_agent.domain.intents import Intent


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Role
    content: str = Field(min_length=1)


class Conversation(BaseModel):
    """The conversation so far. ``customer_id`` comes from a verified token, never from text."""

    model_config = ConfigDict(frozen=True)

    messages: list[Message] = Field(min_length=1)
    customer_id: str | None = None

    @model_validator(mode="after")
    def _last_message_is_from_user(self) -> Self:
        if self.messages[-1].role is not Role.USER:
            raise ValueError("the last message must come from the user")
        return self


class Behavior(StrEnum):
    """What the agent did with the last user turn (docs/evaluation-plan.md §3.2)."""

    ANSWER = "answer"
    CLARIFY = "clarify"
    ACTION = "action"
    REFUSE = "refuse"
    ESCALATE = "escalate"
    NO_ANSWER = "no_answer"


class AgentResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    answer: str
    behavior: Behavior
    intent: Intent | None = None
    sources: list[str] = Field(default_factory=list)
    tool_calls: list[str] = Field(default_factory=list)
    # Why a fallback was used (guardrail / no-answer reason codes); empty for normal answers.
    reasons: list[str] = Field(default_factory=list)
