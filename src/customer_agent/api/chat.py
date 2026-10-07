"""Chat endpoints. Thin adapters: HTTP ⇄ ``ChatService``."""

import json
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from customer_agent.api.schemas import ChatRequest, ChatResponse
from customer_agent.application.answering import TextDelta
from customer_agent.application.chat import ChatService, ChatTurn, SessionNotFoundError
from customer_agent.observability import request_id_var

router = APIRouter(prefix="/v1", tags=["chat"])


def get_chat(request: Request) -> ChatService:
    chat: ChatService = request.app.state.chat
    return chat


Chat = Annotated[ChatService, Depends(get_chat)]
_NOT_FOUND = HTTPException(status_code=404, detail="session not found")


def _response(turn: ChatTurn, request_id: str) -> ChatResponse:
    return ChatResponse(
        session_id=turn.session_id,
        answer=turn.response.answer,
        behavior=turn.response.behavior,
        intent=turn.response.intent,
        sources=turn.response.sources,
        request_id=request_id,
    )


# customer_id is None until authentication arrives (Day 5); it never comes from the body.
@router.post("/chat")
async def chat(body: ChatRequest, service: Chat) -> ChatResponse:
    try:
        turn = await service.chat(body.session_id, None, body.message)
    except SessionNotFoundError:
        raise _NOT_FOUND from None
    return _response(turn, request_id_var.get())


def _sse(event: str, data: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post(
    "/chat/stream",
    response_class=StreamingResponse,
    responses={
        200: {"content": {"text/event-stream": {}}},
        404: {"description": "Unknown session"},
    },
)
async def chat_stream(body: ChatRequest, service: Chat) -> StreamingResponse:
    """Server-Sent Events: ``delta`` events with text, then one authoritative ``final`` event.

    Clients must replace the streamed text with ``final.answer``: an output guardrail may
    substitute a fallback after streaming started.
    """
    request_id = request_id_var.get()
    events = service.chat_stream(body.session_id, None, body.message)
    try:
        first = await anext(events)  # resolve the session before the 200 response starts
    except SessionNotFoundError:
        raise _NOT_FOUND from None

    async def body_events() -> AsyncIterator[str]:
        # The body is produced after the middleware returned; restore the request id for logs.
        token = request_id_var.set(request_id)
        try:
            event: TextDelta | ChatTurn | None = first
            while event is not None:
                if isinstance(event, TextDelta):
                    yield _sse("delta", {"text": event.text})
                else:
                    yield _sse("final", _response(event, request_id).model_dump(mode="json"))
                event = await anext(events, None)
        finally:
            request_id_var.reset(token)

    return StreamingResponse(body_events(), media_type="text/event-stream")
