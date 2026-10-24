"""Chat endpoints. Thin adapters: HTTP ⇄ ``ChatService``."""

import json
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from customer_agent.api.auth import CurrentPrincipal
from customer_agent.api.rate_limit import RateLimited
from customer_agent.api.schemas import ChatRequest, ChatResponse, FeedbackRequest
from customer_agent.application.answering import TextDelta
from customer_agent.application.chat import (
    ChatService,
    ChatTurn,
    FeedbackNotRequestedError,
    SessionNotFoundError,
)
from customer_agent.application.ports import ConcurrentTurnError
from customer_agent.observability import request_id_var

router = APIRouter(prefix="/v1", tags=["chat"])


def get_chat(request: Request) -> ChatService:
    chat: ChatService = request.app.state.chat
    return chat


Chat = Annotated[ChatService, Depends(get_chat)]
_NOT_FOUND = HTTPException(status_code=404, detail="session not found")
_CONFLICT_DETAIL = "another message of this session was processed at the same time; please retry"


def _response(turn: ChatTurn, request_id: str) -> ChatResponse:
    return ChatResponse(
        session_id=turn.session_id,
        answer=turn.response.answer,
        behavior=turn.response.behavior,
        intent=turn.response.intent,
        sources=turn.response.sources,
        request_id=request_id,
        feedback_requested=turn.feedback_requested,
    )


# The customer comes only from a verified token, never from the request body (T06, T13).
@router.post("/chat")
async def chat(
    body: ChatRequest, service: Chat, principal: CurrentPrincipal, _: RateLimited
) -> ChatResponse:
    customer_id = principal.customer_id if principal else None
    try:
        turn = await service.chat(body.session_id, customer_id, body.message)
    except SessionNotFoundError:
        raise _NOT_FOUND from None
    except ConcurrentTurnError:
        raise HTTPException(status_code=409, detail=_CONFLICT_DETAIL) from None
    return _response(turn, request_id_var.get())


def _sse(event: str, data: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post(
    "/chat/stream",
    response_class=StreamingResponse,
    responses={
        200: {"content": {"text/event-stream": {}}},
        401: {"description": "Invalid token"},
        429: {"description": "Too many messages; see Retry-After"},
        404: {"description": "Unknown session"},
        409: {"description": "Concurrent turn on the same session"},
    },
)
async def chat_stream(
    body: ChatRequest, service: Chat, principal: CurrentPrincipal, _: RateLimited
) -> StreamingResponse:
    """Server-Sent Events: ``delta`` events with text, then one authoritative ``final`` event.

    Clients must replace the streamed text with ``final.answer``: an output guardrail may
    substitute a fallback after streaming started.
    """
    request_id = request_id_var.get()
    events = service.chat_stream(
        body.session_id, principal.customer_id if principal else None, body.message
    )
    try:
        # Resolve the session and take its lease before the 200 response starts.
        first = await anext(events)
    except SessionNotFoundError:
        raise _NOT_FOUND from None
    except ConcurrentTurnError:
        raise HTTPException(status_code=409, detail=_CONFLICT_DETAIL) from None

    async def body_events() -> AsyncIterator[str]:
        event: TextDelta | ChatTurn | None = first
        while event is not None:
            if isinstance(event, TextDelta):
                yield _sse("delta", {"text": event.text})
            else:
                yield _sse("final", _response(event, request_id).model_dump(mode="json"))
            event = await anext(events, None)

    return StreamingResponse(body_events(), media_type="text/event-stream")


@router.post(
    "/feedback",
    status_code=204,
    responses={404: {"description": "Unknown session, or it was not asked for feedback"}},
)
async def feedback(
    body: FeedbackRequest, service: Chat, principal: CurrentPrincipal, _: RateLimited
) -> Response:
    """Rate the conversation (👍/👎, optional comment) after a reply with ``feedback_requested``;
    only the session's customer can. A later rating replaces the earlier one."""
    try:
        await service.feedback(
            body.session_id,
            principal.customer_id if principal else None,
            helpful=body.rating == "up",
            comment=body.comment,
        )
    except (SessionNotFoundError, FeedbackNotRequestedError):
        raise HTTPException(status_code=404, detail="no feedback requested") from None
    return Response(status_code=204)
