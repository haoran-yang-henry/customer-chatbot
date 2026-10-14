"""OpenAI-compatible mock for the load test (Day 7): chat completions and embeddings.

The load test measures the system's own overhead (NFR-09), so the model is replaced by this
server with a fixed, configurable latency (``MOCK_LLM_LATENCY_MS``, default 0). It answers the
router, the attack classifier and the answer prompt plausibly enough that every turn takes the
full path: input guardrail, routing, retrieval, answer, output guardrail. Embeddings come from
the same hashing embedder the integration tests use, so a knowledge base ingested through this
mock is searchable through it.

    uvicorn tests.load.mock_llm:app --port 9100
"""

import asyncio
import json
import os
import re
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from customer_agent.infrastructure.embeddings import HashingEmbeddingProvider

LATENCY = float(os.environ.get("MOCK_LLM_LATENCY_MS", "0")) / 1000
_DOCUMENT = re.compile(r'<document id="([^"]+)"')
_EMBEDDER = HashingEmbeddingProvider(dimensions=1536)

app = FastAPI(title="mock-llm")


def _reply(messages: list[dict[str, str]]) -> str:
    system, user = messages[0]["content"], messages[-1]["content"]
    if "You classify customer messages" in system:
        question = user.rsplit("LATEST CUSTOMER MESSAGE:", 1)[-1].strip()
        return json.dumps(
            {"intent": "RETURN_POLICY", "confidence": 0.9, "standalone_query": question}
        )
    if "security filter" in system:
        return json.dumps({"attack_probability": 0.01, "category": "none"})
    cited = _DOCUMENT.search(user)
    source = cited.group(1) if cited else "policy-returns-v5"
    return (
        "Sie können Artikel innerhalb von 30 Tagen zurücksenden, die Rücksendung ist für Sie "
        f"kostenlos. [{source}]"
    )


def _usage(text: str) -> dict[str, int]:
    return {"prompt_tokens": 900, "completion_tokens": len(text) // 4, "total_tokens": 0}


@app.post("/v1/chat/completions", response_model=None)
async def chat(request: Request) -> JSONResponse | StreamingResponse:
    body: dict[str, Any] = await request.json()
    text = _reply(body["messages"])
    await asyncio.sleep(LATENCY)
    base = {"id": "mock", "created": int(time.time()), "model": body["model"]}
    if not body.get("stream"):
        return JSONResponse(
            {
                **base,
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": _usage(text),
            }
        )

    async def events() -> AsyncIterator[str]:
        for start in range(0, len(text), 12):
            delta = {"content": text[start : start + 12]}
            choice = {"index": 0, "delta": delta, "finish_reason": None}
            chunk = {**base, "object": "chat.completion.chunk", "choices": [choice]}
            yield f"data: {json.dumps(chunk)}\n\n"
        final = {**base, "object": "chat.completion.chunk", "choices": [], "usage": _usage(text)}
        yield f"data: {json.dumps(final)}\n\ndata: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/v1/embeddings")
async def embeddings(request: Request) -> JSONResponse:
    body: dict[str, Any] = await request.json()
    texts = body["input"] if isinstance(body["input"], list) else [body["input"]]
    vectors = await _EMBEDDER.embed_documents(texts)
    return JSONResponse(
        {
            "object": "list",
            "model": body["model"],
            "data": [
                {"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vectors)
            ],
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }
    )
