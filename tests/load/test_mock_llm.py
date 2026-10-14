"""The load-test mock speaks the OpenAI protocol as far as the agent uses it."""

import json

import httpx2
from openai import AsyncOpenAI

from customer_agent.application.ports import ChatMessage
from customer_agent.infrastructure.embeddings import OpenAIEmbeddingProvider
from customer_agent.infrastructure.llm import OpenAICompatibleLLM
from customer_agent.infrastructure.prompts import load_prompt
from tests.load.mock_llm import app


def client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key="mock",
        base_url="http://mock/v1",
        http_client=httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app)),
    )


async def test_router_prompt_gets_a_routing_and_answers_cite_the_evidence() -> None:
    llm = OpenAICompatibleLLM(client(), "mock-model")
    router = load_prompt("router", 4)

    routing = await llm.complete(
        [
            ChatMessage(role="system", content=router.system),
            ChatMessage(role="user", content="LATEST CUSTOMER MESSAGE:\nKosten?"),
        ],
        max_tokens=50,
        temperature=0,
    )
    answer = "".join(
        [
            d
            async for d in llm.stream(
                [
                    ChatMessage(role="system", content="You answer."),
                    ChatMessage(role="user", content='<document id="faq-x">…</document>'),
                ],
                max_tokens=50,
                temperature=0,
            )
        ]
    )

    assert json.loads(routing.text)["standalone_query"] == "Kosten?"
    assert answer.endswith("[faq-x]")


async def test_embeddings_have_the_configured_dimensions() -> None:
    embeddings = OpenAIEmbeddingProvider(client(), "mock-embed", dimensions=1536)

    vectors = await embeddings.embed_documents(["Rücksendung", "Versand"])

    assert [len(v) for v in vectors] == [1536, 1536] and vectors[0] != vectors[1]
