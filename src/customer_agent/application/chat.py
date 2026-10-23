"""Chat use case: check the session's owner, take its lease, run one turn.

The client sends only the new message and a ``session_id``; the server owns the transcript and
the explicit state (ADR-0005), both in the turn graph's checkpoint. The session's lease admits
one turn at a time: a concurrent one is rejected before it starts (409).

Each turn runs in its own task, decoupled from the HTTP response: if the client disconnects
while an answer streams, the turn still completes and is checkpointed, so the next turn sees a
consistent transcript and state.

A sampled share of conversations is asked once, after several rounds, whether it helped (the
turn graph decides, ``FeedbackPolicy``). Only an asked session can be rated; the rating is
stored and becomes a score on the session in Langfuse.
"""

import asyncio
from collections.abc import AsyncGenerator

from pydantic import BaseModel, ConfigDict

from customer_agent.application.answering import TextDelta
from customer_agent.application.orchestrator import ConversationOrchestrator, TurnResult
from customer_agent.application.ports import (
    AskedFeedback,
    ConcurrentTurnError,
    SessionStore,
    StoredSession,
)
from customer_agent.domain.conversation import AgentResponse
from customer_agent.domain.pii import mask_personal_data
from customer_agent.observability import (
    current_trace_id,
    get_logger,
    request_id_var,
    score_session,
    session_id_var,
    step,
    trace_turn,
)

log = get_logger("session")


class SessionNotFoundError(LookupError):
    """Unknown session, or one of another customer or tenant (indistinguishable)."""


class FeedbackNotRequestedError(LookupError):
    """The session was never asked whether the conversation helped."""


class ChatTurn(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    response: AgentResponse
    feedback_requested: bool = False  # ask whether the conversation helped


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
                        if event.feedback_requested:
                            await self._ask_feedback(session.session_id)
                        # The checkpoint is written: the client may send the next message as
                        # soon as it has this reply.
                        await self._release(session.session_id)
                        released = True
                        out.put_nowait(
                            ChatTurn(
                                session_id=session.session_id,
                                response=event.response,
                                feedback_requested=event.feedback_requested,
                            )
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

    async def _ask_feedback(self, session_id: str) -> None:
        """Remember the turn that asked, so the rating can be traced back to it."""
        request_id = request_id_var.get()
        asked = AskedFeedback(
            session_id=session_id,
            request_id=None if request_id == "-" else request_id,  # "-": not over HTTP
            trace_id=current_trace_id(),
        )
        with step("session.ask_feedback"):
            await self._sessions.ask_feedback(asked)

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
        *,
        helpful: bool,
        comment: str | None = None,
    ) -> None:
        """The customer rates the conversation they were asked about; a later rating replaces
        it. The comment is masked like any customer text before it is stored or traced
        (NFR-12)."""
        await self._open(session_id, customer_id)  # SessionNotFoundError: unknown or not theirs
        masked = mask_personal_data(comment)[0] if comment else None
        asked = await self._sessions.rate_session(session_id, helpful=helpful, comment=masked)
        if asked is None:
            raise FeedbackNotRequestedError(session_id)
        score_session(session_id, helpful=helpful, comment=masked, asked_in=asked.trace_id)
        log.info("feedback helpful=%s", helpful)

    async def drain(self) -> None:
        """Wait for turns still running (e.g. after client disconnects); call on shutdown."""
        if self._running:
            log.info("draining %d running turn(s)", len(self._running))
            await asyncio.gather(*self._running, return_exceptions=True)
