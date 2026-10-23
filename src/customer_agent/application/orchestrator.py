"""Conversation orchestrator: the ``CustomerServiceAgent`` behind the API and the evaluation.

One turn in two stages (rebuild plan §4.2): the understand agent checks the message and routes
it (input guardrail, rule routing, intent router ‖ attack classifier); the turn graph then
decides and acts on the masked message (``decide``, order guidance, tickets, the grounded
answer) and streams the answer's tokens.

Served sessions keep their transcript and state in the graph's checkpoint (thread = session
id); stateless callers (the evaluation) bring the transcript and get a state rebuilt from it.
"""

import re
from collections.abc import AsyncIterator
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict

from customer_agent.application.answering import (
    AnswerService,
    TextDelta,
    stateless_session_id,
)
from customer_agent.application.fallbacks import TICKET_QUESTION, intent_of_question
from customer_agent.application.feedback import FeedbackPolicy
from customer_agent.application.ports import AuditLog, Clock, OrderRepository, TicketService
from customer_agent.application.tools import CreateTicket, GetOrderStatus
from customer_agent.application.turn_graph import (
    MAX_CONTEXT_MESSAGES,
    TurnContext,
    TurnGraph,
    TurnState,
    turn_input,
)
from customer_agent.application.understanding import UnderstandAgent
from customer_agent.domain.conversation import AgentResponse, Conversation, Message, Role
from customer_agent.domain.routing import extract_order_id
from customer_agent.domain.session import SessionState, WorkflowState
from customer_agent.observability import get_logger, langchain_callbacks, record_guardrail
from customer_agent.observability import step as agent_step

log = get_logger("chat")

_TICKET_NUMBER = re.compile(r"Ticketnummer (?:lautet )?(T-\d{6})")

Graph = CompiledStateGraph[TurnState, TurnContext, TurnState, TurnState]


class TurnResult(BaseModel):
    """The agent's reply, the session state after the turn and the customer message as the
    guardrail left it (personal data masked, never the raw text)."""

    model_config = ConfigDict(frozen=True)

    response: AgentResponse
    state: SessionState
    user_text: str
    feedback_requested: bool = False  # the reply asks whether the conversation helped


TurnEvent = TextDelta | TurnResult


class ConversationOrchestrator:
    def __init__(
        self,
        understand: UnderstandAgent,
        answers: AnswerService,
        *,
        orders: OrderRepository,
        tickets: TicketService,
        audit: AuditLog,
        clock: Clock,
        tenant_id: str,
        checkpointer: BaseCheckpointSaver[Any],
        max_tickets_per_day: int | None = 3,  # None: no limit
        feedback: FeedbackPolicy | None = None,  # None: the defaults
    ) -> None:
        self._understand = understand
        turn = TurnGraph(
            answers,
            GetOrderStatus(orders=orders, audit_log=audit),
            CreateTicket(tickets=tickets, audit_log=audit, max_per_day=max_tickets_per_day),
            clock,
            feedback or FeedbackPolicy(),
        )
        self._stateless = turn.compile()
        self._sessions = turn.compile(checkpointer)
        self._tenant_id = tenant_id

    # --- stateless turns (the evaluation, tests) --------------------------------------------

    async def respond(self, conversation: Conversation) -> AgentResponse:
        turn = await self.handle(
            conversation, SessionState(), session_id=stateless_session_id(conversation)
        )
        return turn.response

    async def handle(
        self, conversation: Conversation, state: SessionState, *, session_id: str
    ) -> TurnResult:
        async for event in self.handle_stream(conversation, state, session_id=session_id):
            if isinstance(event, TurnResult):
                return event
        raise RuntimeError("turn ended without a result")  # pragma: no cover

    async def handle_stream(
        self, conversation: Conversation, state: SessionState, *, session_id: str
    ) -> AsyncIterator[TurnEvent]:
        """The caller brings the transcript; what a session would hold is rebuilt from it."""
        recovered = recover_state(conversation, state)
        async for event in self._turn(
            self._stateless, conversation, recovered, session_id, earlier=conversation.messages[:-1]
        ):
            yield event

    # --- served sessions (ChatService) ------------------------------------------------------

    async def session_stream(
        self, session_id: str, customer_id: str | None, text: str
    ) -> AsyncIterator[TurnEvent]:
        """One turn of a served session; the caller holds the session's lease."""
        config: RunnableConfig = {"configurable": {"thread_id": session_id}}
        messages, state = await self.transcript(session_id)
        history = messages[-(MAX_CONTEXT_MESSAGES - 1) :]
        conversation = Conversation(
            messages=[*history, Message(role=Role.USER, content=text)], customer_id=customer_id
        )
        async for event in self._turn(
            self._sessions, conversation, state, session_id, earlier=[], config=config
        ):
            yield event

    async def transcript(self, session_id: str) -> tuple[list[Message], SessionState]:
        """The masked transcript and the state of a served session, from its checkpoint."""
        snapshot = await self._sessions.aget_state({"configurable": {"thread_id": session_id}})
        values = cast(dict[str, Any], snapshot.values)
        return list(values.get("messages", [])), values.get("session", SessionState())

    async def _turn(
        self,
        graph: Graph,
        conversation: Conversation,
        state: SessionState,
        session_id: str,
        *,
        earlier: list[Message],
        config: RunnableConfig | None = None,
    ) -> AsyncIterator[TurnEvent]:
        """Stage 1, then stage 2 with only the masked message; ``earlier``: what the graph's
        transcript lacks before it (empty when the checkpoint has it). Both stages report to
        the same tracing callbacks; the agents, tools and the retriever inherit them."""
        callbacks: RunnableConfig = {"callbacks": langchain_callbacks()}
        with agent_step("understand"):
            understood = await self._understand.run(
                conversation, state, session_id=session_id, config=callbacks
            )
        record_guardrail(understood.input)
        text = understood.text or "[leere Nachricht]"
        final = turn_input(
            [*earlier, Message(role=Role.USER, content=text)],
            state,
            screen=understood.input,
            routing=understood.routing,
            verdict=understood.verdict,
        )
        context = TurnContext(
            session_id=session_id, tenant_id=self._tenant_id, customer_id=conversation.customer_id
        )
        parts = graph.astream(
            final,
            {**callbacks, **(config or {})},
            context=context,
            stream_mode=["custom", "values"],
            # One checkpoint per turn (rebuild plan §9, phase 0); stateless runs keep none.
            durability="exit" if config is not None else None,
        )
        async for mode, data in cast(AsyncIterator[tuple[str, Any]], parts):
            if mode == "custom":
                yield cast(TextDelta, data)
            else:
                final = cast(TurnState, data)
        response = final["response"]
        if response is None:  # pragma: no cover - the graph always ends with a response
            raise RuntimeError("turn ended without a response")
        yield TurnResult(
            response=response,
            state=final["session"],
            user_text=text,
            feedback_requested=final["feedback_requested"],
        )


def recover_state(conversation: Conversation, state: SessionState) -> SessionState:
    """A stateless call (the evaluation) has the transcript but no session state. Rebuild what
    a session would hold from the earlier messages: the order last talked about (looked up again,
    scoped to the customer, before any use), an open ticket question of the assistant, and a
    ticket already opened. Sessions with state keep theirs."""
    if state != SessionState() or len(conversation.messages) < 2:
        return state
    earlier = conversation.messages[:-1]
    slots: dict[str, str] = {}
    for message in earlier:
        if order_id := extract_order_id(message.content):
            slots["order_id"] = order_id
        if message.role is Role.ASSISTANT and (match := _TICKET_NUMBER.search(message.content)):
            slots["ticket_id"] = match.group(1)
    last = earlier[-1]
    if last.role is not Role.ASSISTANT or TICKET_QUESTION not in last.content:
        return SessionState(slots=slots) if slots else state
    question = last.content[last.content.rfind("Soll ich") :]
    slots["question"] = question
    if order_id := extract_order_id(question):
        slots["order_id"] = order_id
    requests = [m.content for m in earlier[:-1] if m.role is Role.USER][-3:]
    if requests:
        slots["request"] = " / ".join(requests)
    return SessionState(
        active_intent=intent_of_question(question),
        workflow=WorkflowState.AWAITING_TICKET_CONFIRMATION,
        slots=slots,
    )
