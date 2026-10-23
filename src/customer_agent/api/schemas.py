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


class FeedbackRequest(BaseModel):
    """The customer's rating of one answer of their session."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"session_id": "4f1c…", "request_id": "33e7c505b6ef", "rating": "down"}]
        }
    )

    session_id: str = Field(min_length=1, max_length=64)
    request_id: str = Field(
        min_length=1, max_length=64, description="The ``request_id`` of the rated answer."
    )
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=1000)
