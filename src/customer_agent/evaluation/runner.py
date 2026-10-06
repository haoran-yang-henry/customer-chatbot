"""Runs a dataset against an agent and aggregates results into a report."""

import asyncio
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, computed_field

from customer_agent.application.ports import CustomerServiceAgent
from customer_agent.evaluation.schema import EvalCase
from customer_agent.evaluation.scoring import CaseResult, score_case


class RateSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    passed: int
    total: int

    @computed_field  # type: ignore[prop-decorator]
    @property
    def rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


class EvalReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    target: str
    dataset: str
    current_day: int
    started_at: datetime
    config: dict[str, str] = {}
    overall: RateSummary
    gated: RateSummary
    by_category: dict[str, RateSummary]
    by_check: dict[str, RateSummary]
    groundedness: RateSummary | None = None
    results: list[CaseResult]


def _summarise(flags: Sequence[bool]) -> RateSummary:
    return RateSummary(passed=sum(flags), total=len(flags))


def build_report(
    results: list[CaseResult],
    *,
    target: str,
    dataset: str,
    current_day: int,
    started_at: datetime,
    config: dict[str, str] | None = None,
) -> EvalReport:
    by_category: defaultdict[str, list[bool]] = defaultdict(list)
    by_check: defaultdict[str, list[bool]] = defaultdict(list)
    for result in results:
        by_category[result.category].append(result.passed)
        for check, ok in result.checks.items():
            by_check[check].append(ok)
    judged = [r.grounded for r in results if r.grounded is not None]
    return EvalReport(
        target=target,
        dataset=dataset,
        current_day=current_day,
        started_at=started_at,
        config=config or {},
        overall=_summarise([r.passed for r in results]),
        gated=_summarise([r.passed for r in results if r.target_day <= current_day]),
        by_category={k: _summarise(v) for k, v in sorted(by_category.items())},
        by_check={k: _summarise(v) for k, v in sorted(by_check.items())},
        groundedness=_summarise(judged) if judged else None,
        results=results,
    )


async def run_cases(
    cases: Sequence[EvalCase],
    agent: CustomerServiceAgent,
    *,
    current_day: int,
    concurrency: int = 4,
) -> list[CaseResult]:
    """Results keep dataset order; ``concurrency`` bounds parallel calls to the agent."""
    semaphore = asyncio.Semaphore(concurrency)

    async def one(case: EvalCase) -> CaseResult:
        async with semaphore:
            response = await agent.respond(case.to_conversation())
        return score_case(case, response, current_day)

    return list(await asyncio.gather(*(one(c) for c in cases)))


async def run_evaluation(
    cases: list[EvalCase],
    agent: CustomerServiceAgent,
    *,
    target: str,
    dataset: str,
    current_day: int,
    config: dict[str, str] | None = None,
    concurrency: int = 4,
) -> EvalReport:
    started_at = datetime.now(UTC)
    results = await run_cases(cases, agent, current_day=current_day, concurrency=concurrency)
    return build_report(
        results,
        target=target,
        dataset=dataset,
        current_day=current_day,
        started_at=started_at,
        config=config,
    )
