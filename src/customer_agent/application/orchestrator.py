"""Conversation orchestrator: the ``CustomerServiceAgent`` behind the API and the evaluation.

One turn: input guardrail → intent router → state machine (``decide``) → handler. Order
guidance with real order data (Day 5) and full guardrails (Day 6) plug in here.
"""

from collections.abc import AsyncIterator

from pydantic import BaseModel, ConfigDict

from customer_agent.application.answering import (
    AnswerService,
    FinalAnswer,
    TextDelta,
)
from customer_agent.application.answering import (
    session_id as conversation_hash,
)
from customer_agent.application.fallbacks import fallback, step_reply
from customer_agent.application.guardrails import check_input
from customer_agent.application.ports import ProviderUnavailableError
from customer_agent.application.routing import IntentRouter
from customer_agent.domain.conversation import AgentResponse, Conversation
from customer_agent.domain.intents import Intent
from customer_agent.domain.routing import RoutingResult
from customer_agent.domain.session import Decision, NextStep, SessionState, decide
from customer_agent.observability import get_logger

log = get_logger("chat")


class TurnResult(BaseModel):
    """The agent's reply plus the session state to persist for the next turn."""

    model_config = ConfigDict(frozen=True)

    response: AgentResponse
    state: SessionState


TurnEvent = TextDelta | TurnResult


class ConversationOrchestrator:
    def __init__(self, router: IntentRouter, answers: AnswerService) -> None:
        self._router = router
        self._answers = answers

    # --- CustomerServiceAgent (stateless; used by the evaluation) -------------------------

    async def respond(self, conversation: Conversation) -> AgentResponse:
        turn = await self.handle(
            conversation, SessionState(), session_id=conversation_hash(conversation)
        )
        return turn.response

    # --- stateful turns (used by ChatService) -----------------------------------------------

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
        decision = check_input(conversation.messages[-1].content)
        if decision.blocked:
            log.info("input blocked reasons=%s", ",".join(decision.reasons))
            yield TurnResult(response=fallback(decision.reasons), state=state)
            return
        try:
            routing = await self._router.route(conversation, state, session_id=session_id)
            step = decide(state, routing, authenticated=conversation.customer_id is not None)
            log.info("step=%s intent=%s workflow=%s", step.step, step.intent, step.state.workflow)
            async for event in self._execute(conversation, routing, step, session_id):
                yield event
        except ProviderUnavailableError as exc:
            log.error("provider unavailable: %s", exc)
            yield TurnResult(response=fallback(["provider_unavailable"]), state=state)

    async def _execute(
        self,
        conversation: Conversation,
        routing: RoutingResult,
        step: Decision,
        session_id: str,
    ) -> AsyncIterator[TurnEvent]:
        def result(response: AgentResponse) -> TurnResult:
            final = response.model_copy(update={"intent": step.intent})
            log.info("behavior=%s sources=%s", final.behavior, ",".join(final.sources) or "-")
            return TurnResult(response=final, state=step.state)

        match step.step:
            case NextStep.ANSWER_FROM_KNOWLEDGE | NextStep.ORDER_GUIDANCE:
                # Day 4: order guidance answers from policy knowledge; Day 5 adds order data.
                async for event in self._answers.answer_stream(
                    conversation, query=routing.standalone_query, session_id=session_id
                ):
                    if isinstance(event, FinalAnswer):
                        yield result(event.response)
                    else:
                        yield event
            case NextStep.HANDOFF:
                key = "handoff_complaint" if step.intent is Intent.COMPLAINT else "handoff"
                yield result(step_reply(key))
            case NextStep.CLARIFY:
                yield result(step_reply("clarify", text=routing.clarification_question))
            case _:
                yield result(step_reply(step.step.value))
