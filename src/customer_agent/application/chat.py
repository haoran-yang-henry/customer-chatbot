"""Chat use case: load the session, run one orchestrated turn, persist transcript and state.

The client sends only the new message and a ``session_id``; the server owns the transcript and
the explicit state (ADR-0005).
"""

from collections.abc import AsyncIterator

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
    """Unknown session, or a session that belongs to another customer (indistinguishable)."""


class ChatTurn(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    response: AgentResponse


class ChatService:
    def __init__(
        self, orchestrator: ConversationOrchestrator, sessions: SessionStore, *, tenant_id: str
    ) -> None:
        self._orchestrator = orchestrator
        self._sessions = sessions
        self._tenant_id = tenant_id

    async def _open(self, session_id: str | None, customer_id: str | None) -> StoredSession:
        if session_id is None:
            session = await self._sessions.create(
                tenant_id=self._tenant_id, customer_id=customer_id
            )
            log.info("created session=%s", session.session_id)
            return session
        loaded = await self._sessions.load(session_id)
        if loaded is None or loaded.customer_id != customer_id:
            raise SessionNotFoundError(session_id)
        return loaded

    async def chat_stream(
        self, session_id: str | None, customer_id: str | None, text: str
    ) -> AsyncIterator[TextDelta | ChatTurn]:
        session = await self._open(session_id, customer_id)
        history = session.messages[-(MAX_CONTEXT_MESSAGES - 1) :]
        conversation = Conversation(
            messages=[*history, Message(role=Role.USER, content=text)], customer_id=customer_id
        )
        async for event in self._orchestrator.handle_stream(
            conversation, session.state, session_id=session.session_id
        ):
            if isinstance(event, TurnResult):
                await self._sessions.save_turn(
                    session.session_id, state=event.state, user=text, reply=event.response
                )
                yield ChatTurn(session_id=session.session_id, response=event.response)
            else:
                yield event

    async def chat(self, session_id: str | None, customer_id: str | None, text: str) -> ChatTurn:
        async for event in self.chat_stream(session_id, customer_id, text):
            if isinstance(event, ChatTurn):
                return event
        raise RuntimeError("turn ended without a result")  # pragma: no cover
