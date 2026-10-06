"""Conversation orchestrator: the ``CustomerServiceAgent`` behind the API and the evaluation.

Day 3: input guardrail → grounded answer. Intent routing (Day 4), order guidance (Day 5) and
full guardrails (Day 6) plug in here.
"""

from collections.abc import AsyncIterator

from customer_agent.application.answering import AnswerService, FinalAnswer, StreamEvent
from customer_agent.application.fallbacks import fallback
from customer_agent.application.guardrails import check_input
from customer_agent.application.ports import ProviderUnavailableError
from customer_agent.domain.conversation import AgentResponse, Conversation
from customer_agent.observability import get_logger

log = get_logger("chat")


class ConversationOrchestrator:
    def __init__(self, answers: AnswerService) -> None:
        self._answers = answers

    def _blocked_input(self, conversation: Conversation) -> AgentResponse | None:
        decision = check_input(conversation.messages[-1].content)
        if decision.blocked:
            log.info("input blocked reasons=%s", ",".join(decision.reasons))
            return fallback(decision.reasons)
        return None

    async def respond(self, conversation: Conversation) -> AgentResponse:
        if blocked := self._blocked_input(conversation):
            return blocked
        try:
            response = await self._answers.answer(conversation)
        except ProviderUnavailableError as exc:
            log.error("provider unavailable: %s", exc)
            return fallback(["provider_unavailable"])
        log.info("behavior=%s sources=%s", response.behavior, ",".join(response.sources) or "-")
        return response

    async def respond_stream(self, conversation: Conversation) -> AsyncIterator[StreamEvent]:
        if blocked := self._blocked_input(conversation):
            yield FinalAnswer(response=blocked)
            return
        try:
            async for event in self._answers.answer_stream(conversation):
                if isinstance(event, FinalAnswer):
                    log.info(
                        "behavior=%s sources=%s",
                        event.response.behavior,
                        ",".join(event.response.sources) or "-",
                    )
                yield event
        except ProviderUnavailableError as exc:
            log.error("provider unavailable: %s", exc)
            yield FinalAnswer(response=fallback(["provider_unavailable"]))
