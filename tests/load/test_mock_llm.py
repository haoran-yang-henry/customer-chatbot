"""The load-test mock speaks the OpenAI protocol as far as the agent uses it."""

import json

import httpx2
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import SecretStr

from customer_agent.infrastructure.embeddings import ProviderEmbeddings
from customer_agent.infrastructure.models import ProviderChatModel
from customer_agent.infrastructure.prompts import load_prompt
from tests.load.mock_llm import app


async def test_router_prompt_gets_a_routing_and_answers_cite_the_evidence() -> None:
    llm = ProviderChatModel(
        model_name="mock-model",
        openai_api_key=SecretStr("mock"),
        openai_api_base="http://mock/v1",
        http_async_client=httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app)),
        stream_usage=True,
    )
    router = load_prompt("router", 5)

    routing = await llm.ainvoke(
        [SystemMessage(router.system), HumanMessage("LATEST CUSTOMER MESSAGE:\nKosten?")],
        max_tokens=50,
        temperature=0,
    )
    answer = "".join(
        [
            chunk.text
            async for chunk in llm.astream(
                [SystemMessage("You answer."), HumanMessage('<document id="faq-x">…</document>')],
                max_tokens=50,
                temperature=0,
            )
        ]
    )

    assert json.loads(routing.text)["standalone_query"] == "Kosten?"
    assert answer.endswith("[faq-x]")


async def test_embeddings_have_the_configured_dimensions() -> None:
    embeddings = ProviderEmbeddings(
        model="mock-embed",
        dimensions=1536,
        openai_api_key=SecretStr("mock"),
        openai_api_base="http://mock/v1",
        http_async_client=httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app)),
    )

    vectors = await embeddings.aembed_documents(["Rücksendung", "Versand"])
    await embeddings.aclose()

    assert [len(v) for v in vectors] == [1536, 1536] and vectors[0] != vectors[1]
