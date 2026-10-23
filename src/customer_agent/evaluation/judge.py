"""LLM-as-judge for groundedness (evaluation plan §5).

Judges whether every factual claim of an answer is supported by the documents it cites. The
judge's numbers are only trusted after calibration against human labels (``calibration_rows``,
``agreement``).
"""

import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, ConfigDict

from customer_agent.application.prompts import PromptTemplate
from customer_agent.domain.knowledge import KnowledgeDocument
from customer_agent.evaluation.scoring import CaseResult

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class JudgeVerdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    grounded: bool | None  # None: the judge's output could not be parsed
    unsupported_claims: list[str] = []


def parse_verdict(text: str) -> JudgeVerdict:
    match = _JSON_OBJECT.search(text)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    grounded = data.get("grounded")
    claims = data.get("unsupported_claims") or []
    return JudgeVerdict(
        grounded=grounded if isinstance(grounded, bool) else None,
        unsupported_claims=[str(c) for c in claims] if isinstance(claims, list) else [],
    )


class GroundednessJudge:
    def __init__(
        self, llm: BaseChatModel, prompt: PromptTemplate, documents: Sequence[KnowledgeDocument]
    ) -> None:
        self._llm = llm
        self._prompt = prompt
        self._documents = {d.meta.id: d for d in documents}

    async def judge(self, answer: str, sources: Sequence[str]) -> JudgeVerdict:
        cited = [self._documents[s] for s in sources if s in self._documents]
        text = "\n\n".join(f'<document id="{d.meta.id}">\n{d.body}\n</document>' for d in cited)
        reply = await self._llm.ainvoke(
            self._prompt.render_messages(sources=text, answer=answer),
            session_id="judge-" + hashlib.sha256(answer.encode()).hexdigest()[:24],
            max_tokens=600,
            temperature=0.0,
        )
        return parse_verdict(reply.text)


def judgeable(result: CaseResult) -> bool:
    """Knowledge answers only. Answers built from tool results (order guidance) contain order
    facts the judge cannot see in the cited policies; those templates are verified by
    deterministic unit tests instead (tests/unit/application/test_order_guidance.py)."""
    response = result.response
    return response.behavior == "answer" and bool(response.sources) and not response.tool_calls


def calibration_rows(
    results: Sequence[CaseResult], questions: dict[str, str]
) -> list[dict[str, object]]:
    """Rows for a human to label: fill ``human_grounded`` with true/false."""
    return [
        {
            "case_id": r.case_id,
            "question": questions[r.case_id],
            "answer": r.response.answer,
            "sources": r.response.sources,
            "judge_grounded": r.grounded,
            "human_grounded": None,
        }
        for r in results
        if r.grounded is not None
    ]


def agreement(path: Path) -> tuple[int, int]:
    """(agreeing, labelled) over rows where a human label exists."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    labelled = [r for r in rows if isinstance(r.get("human_grounded"), bool)]
    agreeing = sum(1 for r in labelled if r["human_grounded"] == r["judge_grounded"])
    return agreeing, len(labelled)
