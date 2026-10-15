"""NFR-11 by fault injection: the model provider stops answering. Every turn must reach the
fallback within 10 s without a 5xx, and once the breaker opens the provider is not called."""

import time

import httpx2
from openai import AsyncOpenAI

from customer_agent.application.answering import AnswerService
from customer_agent.application.orchestrator import ConversationOrchestrator
from customer_agent.application.routing import IntentRouter
from customer_agent.domain.conversation import Behavior, Conversation, Message, Role
from customer_agent.infrastructure.llm import OpenAICompatibleLLM
from customer_agent.infrastructure.memory import InMemoryAuditLog, InMemoryTicketService
from customer_agent.infrastructure.prompts import load_prompt
from customer_agent.infrastructure.resilience import BreakerLLM, CircuitBreaker
from tests.fakes import FakeRetriever, FixedClock, SeedOrderRepository, document
from tests.unit.application.test_answering import TODAY

NFR_11_SECONDS = 10.0


async def test_llm_outage_reaches_the_fallback_in_time_and_the_breaker_takes_over() -> None:
    attempts: list[str] = []

    def hanging(request: httpx2.Request) -> httpx2.Response:
        attempts.append(request.url.path)
        raise httpx2.ConnectTimeout("provider unreachable", request=request)

    client = AsyncOpenAI(
        api_key="test",
        base_url="https://llm.test/v1",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(hanging)),
        timeout=0.5,
        max_retries=1,
    )
    llm = BreakerLLM(
        OpenAICompatibleLLM(client, "model-down"),
        CircuitBreaker("fault-llm", failure_threshold=2, reset_after=60),
    )
    agent = ConversationOrchestrator(
        IntentRouter(llm, load_prompt("router", 5)),
        AnswerService(
            FakeRetriever([document("policy-returns-v5", 0.7)]),
            llm,
            load_prompt("answer", 4),
            FixedClock(TODAY),
            tenant_id="nordelektro",
            min_similarity=0.35,
        ),
        orders=SeedOrderRepository(),
        tickets=InMemoryTicketService(),
        audit=InMemoryAuditLog(),
        clock=FixedClock(TODAY),
        tenant_id="nordelektro",
    )

    durations, behaviors = [], []
    for n in range(4):
        start = time.perf_counter()
        message = Message(role=Role.USER, content=f"Frage {n}")
        response = await agent.respond(Conversation(messages=[message], customer_id="C-1001"))
        durations.append(time.perf_counter() - start)
        behaviors.append(response.behavior)
        assert response.tool_calls == []  # contact channels; no ticket for a request not understood

    assert behaviors == [Behavior.ESCALATE] * 4  # never an exception
    assert max(durations) < NFR_11_SECONDS
    assert len(attempts) == 4  # two turns × (call + one retry); then the breaker answers
    assert max(durations[2:]) < 0.1
