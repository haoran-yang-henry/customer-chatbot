"""On-disk response cache for evaluation runs: identical requests are answered from disk.

Only for development and evaluation (``CA_LLM_CACHE_DIR``); never enable it in production —
answers would not reflect knowledge or prompt changes outside the request itself.
"""

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

from customer_agent.application.ports import ChatMessage, LLMProvider, LLMResult
from customer_agent.infrastructure.lifecycle import SupportsAclose
from customer_agent.observability import record_tokens


class CachingLLM:
    """``namespace`` must identify everything besides the request that changes the output
    (endpoint, model, reasoning settings); otherwise a config change silently reuses old answers.
    """

    def __init__(self, inner: LLMProvider, directory: Path, *, namespace: str) -> None:
        self._inner = inner
        self._directory = directory
        self._namespace = namespace
        directory.mkdir(parents=True, exist_ok=True)

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    def _path(self, messages: Sequence[ChatMessage], max_tokens: int, temperature: float) -> Path:
        key = json.dumps(
            [self._namespace, [m.model_dump() for m in messages], max_tokens, temperature],
            ensure_ascii=False,
        )
        return self._directory / f"{hashlib.sha256(key.encode()).hexdigest()}.json"

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
        session_id: str | None = None,
    ) -> LLMResult:
        path = self._path(messages, max_tokens, temperature)
        # File I/O blocks; run it in the default thread pool so concurrent turns keep running.
        if await asyncio.to_thread(path.exists):
            cached = LLMResult.model_validate_json(
                await asyncio.to_thread(path.read_text, encoding="utf-8")
            )
            # Count the recorded usage: an evaluation's token totals are what the run would cost.
            record_tokens(self._inner.model_name, cached.prompt_tokens, cached.completion_tokens)
            return cached
        result = await self._inner.complete(
            messages, max_tokens=max_tokens, temperature=temperature, session_id=session_id
        )
        if result.text.strip():  # never cache an empty (failed) generation
            await asyncio.to_thread(path.write_text, result.model_dump_json(), encoding="utf-8")
        return result

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
        session_id: str | None = None,
    ) -> AsyncIterator[str]:
        result = await self.complete(
            messages, max_tokens=max_tokens, temperature=temperature, session_id=session_id
        )
        yield result.text

    async def aclose(self) -> None:
        if isinstance(self._inner, SupportsAclose):
            await self._inner.aclose()
