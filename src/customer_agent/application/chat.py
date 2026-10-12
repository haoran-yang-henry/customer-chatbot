"""Chat use case: load the session, run one orchestrated turn, persist transcript and state.

The client sends only the new message and a ``session_id``; the server owns the transcript and
the explicit state (ADR-0005).

Each turn runs in its own task, decoupled from the HTTP response: if the client disconnects
while an answer streams, the turn still completes and is saved, so the next turn sees a
consistent transcript and state.
"""

import asyncio
from collections.abc import AsyncGenerator

from pydantic import BaseModel, ConfigDict

from customer_agent.application.answering import TextDelta
from customer_agent.application.orchestrator import ConversationOrchestrator, TurnResult
from customer_agent.application.ports import SessionStore, StoredSession
from customer_agent.domain.conversation import AgentResponse, Conversation, Message, Role
from customer_agent.observability import get_logger

log = get_logger("session")

# Transcript passed to the router and the answer prompt; older turns are kept in the database.
MAX_CONTEXT_MESSAGES = 20


class SessionNotFoundError(LookupError):
    """Unknown session, or one of another customer or tenant (indistinguishable)."""


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
            session = await self._sessions.create(
                tenant_id=self._tenant_id, customer_id=customer_id
            )
            log.info("created session=%s", session.session_id)
            return session
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
        try:
            history = session.messages[-(MAX_CONTEXT_MESSAGES - 1) :]
            conversation = Conversation(
                messages=[*history, Message(role=Role.USER, content=text)],
                customer_id=customer_id,
            )
            async for event in self._orchestrator.handle_stream(
                conversation, session.state, session_id=session.session_id
            ):
                if isinstance(event, TurnResult):
                    await self._sessions.save_turn(
                        session.session_id,
                        expected_version=session.version,
                        state=event.state,
                        user=text,
                        reply=event.response,
                    )
                    out.put_nowait(ChatTurn(session_id=session.session_id, response=event.response))
                else:
                    out.put_nowait(event)
            out.put_nowait(None)
        except Exception as exc:  # handed to the consumer, re-raised there
            log.warning("turn failed: %s", type(exc).__name__)
            out.put_nowait(exc)

    async def chat_stream(
        self, session_id: str | None, customer_id: str | None, text: str
    ) -> AsyncGenerator[ChatEvent, None]:
        session = await self._open(session_id, customer_id)  # errors surface before streaming
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

    async def drain(self) -> None:
        """Wait for turns still running (e.g. after client disconnects); call on shutdown."""
        if self._running:
            log.info("draining %d running turn(s)", len(self._running))
            await asyncio.gather(*self._running, return_exceptions=True)
