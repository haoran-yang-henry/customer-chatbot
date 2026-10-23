"""HTTP request/response models (the public API contract, see docs/api.md)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from customer_agent.domain.conversation import Behavior
from customer_agent.domain.intents import Intent


class ChatRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"examples": [{"message": "Kostet die Rücksendung etwas?"}]}
    )

    session_id: str | None = Field(
        default=None, description="Omit to start a new conversation; then send the returned id."
    )
    message: str = Field(min_length=1, max_length=10_000)


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    behavior: Behavior
    intent: Intent | None
    sources: list[str]
    request_id: str
    feedback_requested: bool = Field(
        default=False,
        description="Ask the customer whether the conversation helped (POST /v1/feedback).",
    )


class FeedbackRequest(BaseModel):
    """The customer's rating of a conversation they were asked about (``feedback_requested``)."""

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"session_id": "4f1c…", "rating": "down"}]}
    )

    session_id: str = Field(min_length=1, max_length=64)
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=1000)
