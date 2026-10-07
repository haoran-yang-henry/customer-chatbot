"""HTTP request/response models (the public API contract, see docs/api.md)."""

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
