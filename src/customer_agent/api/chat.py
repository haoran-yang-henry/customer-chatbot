"""Chat endpoints. Thin adapters: HTTP ⇄ ``Conversation`` / ``AgentResponse``."""

import json
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from customer_agent.api.schemas import ChatRequest, ChatResponse
from customer_agent.application.answering import FinalAnswer, TextDelta
from customer_agent.application.orchestrator import ConversationOrchestrator
from customer_agent.domain.conversation import Conversation
from customer_agent.observability import request_id_var

router = APIRouter(prefix="/v1", tags=["chat"])


def get_agent(request: Request) -> ConversationOrchestrator:
    agent: ConversationOrchestrator = request.app.state.agent
    return agent


Agent = Annotated[ConversationOrchestrator, Depends(get_agent)]


def _conversation(body: ChatRequest) -> Conversation:
    # customer_id comes from the verified token on Day 5, never from the request body.
    try:
        return Conversation(messages=body.messages, customer_id=None)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=exc.errors(include_url=False, include_input=False, include_context=False),
        ) from exc


@router.post("/chat")
async def chat(body: ChatRequest, agent: Agent) -> ChatResponse:
    response = await agent.respond(_conversation(body))
    return ChatResponse.from_agent(response, request_id_var.get())


def _sse(event: str, data: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post(
    "/chat/stream",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}}},
)
async def chat_stream(body: ChatRequest, agent: Agent) -> StreamingResponse:
    """Server-Sent Events: ``delta`` events with text, then one authoritative ``final`` event.

    Clients must replace the streamed text with ``final.answer``: an output guardrail may
    substitute a fallback after streaming started.
    """
    conversation = _conversation(body)
    request_id = request_id_var.get()

    async def events() -> AsyncIterator[str]:
        # The body is produced after the middleware has returned; restore the request id so
        # log lines written while streaming keep it.
        token = request_id_var.set(request_id)
        try:
            async for event in agent.respond_stream(conversation):
                if isinstance(event, TextDelta):
                    yield _sse("delta", {"text": event.text})
                elif isinstance(event, FinalAnswer):
                    final = ChatResponse.from_agent(event.response, request_id)
                    yield _sse("final", final.model_dump(mode="json"))
        finally:
            request_id_var.reset(token)

    return StreamingResponse(events(), media_type="text/event-stream")
