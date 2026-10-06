"""HTTP request/response models (the public API contract, see docs/api.md)."""

from pydantic import BaseModel, ConfigDict, Field

from customer_agent.domain.conversation import AgentResponse, Behavior, Message


class ChatRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"messages": [{"role": "user", "content": "Kostet die Rücksendung etwas?"}]}
            ]
        }
    )

    messages: list[Message] = Field(min_length=1, max_length=50)


class ChatResponse(BaseModel):
    answer: str
    behavior: Behavior
    sources: list[str]
    request_id: str

    @classmethod
    def from_agent(cls, response: AgentResponse, request_id: str) -> "ChatResponse":
        return cls(
            answer=response.answer,
            behavior=response.behavior,
            sources=response.sources,
            request_id=request_id,
        )
