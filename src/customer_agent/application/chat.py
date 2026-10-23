"""Chat use case: check the session's owner, take its lease, run one turn.

The client sends only the new message and a ``session_id``; the server owns the transcript and
the explicit state (ADR-0005), both in the turn graph's checkpoint. The session's lease admits
one turn at a time: a concurrent one is rejected before it starts (409).

Each turn runs in its own task, decoupled from the HTTP response: if the client disconnects
while an answer streams, the turn still completes and is checkpointed, so the next turn sees a
consistent transcript and state.

Every answer can be rated by the session's customer (feedback): the rating is stored with the
turn and becomes a score on the turn's trace in Langfuse.
"""

import asyncio
from collections.abc import AsyncGenerator

from pydantic import BaseModel, ConfigDict

from customer_agent.application.answering import TextDelta
from customer_agent.application.orchestrator import ConversationOrchestrator, TurnResult
from customer_agent.application.ports import (
    ConcurrentTurnError,
    SessionStore,
    StoredSession,
    StoredTurn,
)
from customer_agent.domain.conversation import AgentResponse
from customer_agent.domain.pii import mask_personal_data
from customer_agent.observability import (
    current_trace_id,
    get_logger,
    request_id_var,
    score_turn,
    session_id_var,
    step,
    trace_turn,
)

log = get_logger("session")


class SessionNotFoundError(LookupError):
    """Unknown session, or one of another customer or tenant (indistinguishable)."""


class TurnNotFoundError(LookupError):
    """The session has no answer with this request id."""


class ChatTurn(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    response: AgentResponse


ChatEvent = TextDelta | ChatTurn
_Item = ChatEvent | BaseException | None  # None marks the end of the turn


class ChatService:
    def __init__(
        self, orchestrator: ConversationOrchestrator, sessions: SessionStore, *, tenant_id: str
    ) -> None:
        self._orchestrator = orchestrator
        self._sessions = sessions
        self._tenant_id = tenant_id
        self._running: set[asyncio.Task[None]] = set()

    async def _open(self, session_id: str | None, customer_id: str | None) -> StoredSession:
        if session_id is None:
            with step("session.create"):
                session = await self._sessions.create(
                    tenant_id=self._tenant_id, customer_id=customer_id
                )
            log.info("created session=%s", session.session_id)
            return session
        with step("session.load"):
            loaded = await self._sessions.load(session_id)
        if (
            loaded is None
            or loaded.tenant_id != self._tenant_id
            or loaded.customer_id != customer_id
        ):
            raise SessionNotFoundError(session_id)
        return loaded

    async def _run_turn(
        self,
        session: StoredSession,
        customer_id: str | None,
        text: str,
        out: "asyncio.Queue[_Item]",
    ) -> None:
        """Producer: runs to completion even when nobody reads ``out`` any more."""
        # The task has its own copy of the context: every log line of the turn gets the session.
        session_id_var.set(session.session_id)
        released = False
        try:
            with trace_turn(session_id=session.session_id, customer_id=customer_id) as turn:
                async for event in self._orchestrator.session_stream(
                    session.session_id, customer_id, text
                ):
                    if isinstance(event, TurnResult):
                        turn.record(
                            user_text=event.user_text,  # masked by the input guardrail
                            answer=event.response.answer,
                            metadata={
                                "behavior": event.response.behavior.value,
                                "intent": str(event.response.intent),
                            },
                        )
                        await self._record(session.session_id)
                        # The checkpoint is written: the client may send the next message as
                        # soon as it has this reply.
                        await self._release(session.session_id)
                        released = True
                        out.put_nowait(
                            ChatTurn(session_id=session.session_id, response=event.response)
                        )
                    else:
                        out.put_nowait(event)
            out.put_nowait(None)
        except Exception as exc:  # handed to the consumer, re-raised there
            log.warning("turn failed: %s", type(exc).__name__)
            out.put_nowait(exc)
        finally:
            if not released:
                await self._release(session.session_id)

    async def _record(self, session_id: str) -> None:
        """A turn served over HTTP can be rated later by its request id."""
        request_id = request_id_var.get()
        if request_id == "-":  # not an HTTP request: nothing to refer to
            return
        turn = StoredTurn(session_id=session_id, request_id=request_id, trace_id=current_trace_id())
        with step("session.record_turn"):
            await self._sessions.record_turn(turn)

    async def _release(self, session_id: str) -> None:
        with step("session.release"):
            await self._sessions.release(session_id)

    async def chat_stream(
        self, session_id: str | None, customer_id: str | None, text: str
    ) -> AsyncGenerator[ChatEvent, None]:
        session = await self._open(session_id, customer_id)  # errors surface before streaming
        with step("session.lease"):
            leased = await self._sessions.acquire(session.session_id)
        if not leased:
            raise ConcurrentTurnError(session.session_id)
        out: asyncio.Queue[_Item] = asyncio.Queue()
        task = asyncio.create_task(self._run_turn(session, customer_id, text, out))
        self._running.add(task)
        task.add_done_callback(self._running.discard)
        while (item := await out.get()) is not None:
            if isinstance(item, BaseException):
                raise item
            yield item

    async def chat(self, session_id: str | None, customer_id: str | None, text: str) -> ChatTurn:
        async for event in self.chat_stream(session_id, customer_id, text):
            if isinstance(event, ChatTurn):
                return event
        raise RuntimeError("turn ended without a result")  # pragma: no cover

    async def feedback(
        self,
        session_id: str,
        customer_id: str | None,
        request_id: str,
        *,
        helpful: bool,
        comment: str | None = None,
    ) -> None:
        """The customer rates one answer of their session; a later rating replaces it. The
        comment is masked like any customer text before it is stored or traced (NFR-12)."""
        await self._open(session_id, customer_id)  # SessionNotFoundError: unknown or not theirs
        masked = mask_personal_data(comment)[0] if comment else None
        turn = await self._sessions.rate_turn(
            session_id, request_id, helpful=helpful, comment=masked
        )
        if turn is None:
            raise TurnNotFoundError(request_id)
        if turn.trace_id is not None:
            score_turn(turn.trace_id, helpful=helpful, comment=masked)
        log.info("feedback request_id=%s helpful=%s", request_id, helpful)

    async def drain(self) -> None:
        """Wait for turns still running (e.g. after client disconnects); call on shutdown."""
        if self._running:
            log.info("draining %d running turn(s)", len(self._running))
            await asyncio.gather(*self._running, return_exceptions=True)
