"""Runs a dataset against an agent and aggregates results into a report."""

import asyncio
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, computed_field

from customer_agent.application.ports import CustomerServiceAgent
from customer_agent.domain.conversation import Behavior
from customer_agent.domain.guardrails import BLOCK_REASONS, REDACTION_REASONS
from customer_agent.evaluation.schema import Category, EvalCase
from customer_agent.evaluation.scoring import CaseResult, score_case


class RateSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    passed: int
    total: int

    @computed_field  # type: ignore[prop-decorator]
    @property
    def rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


class SafetySummary(BaseModel):
    """Day 6 metrics (NFR-07, NFR-08). An attack succeeds when an adversarial case fails any
    check; a false positive is a benign case answered with a guardrail block."""

    model_config = ConfigDict(frozen=True)

    attacks: int
    successful_attacks: int
    benign: int
    blocked_benign: int
    redacted_benign: int  # delivered, but with sentences removed or values masked

    @computed_field  # type: ignore[prop-decorator]
    @property
    def attack_success_rate(self) -> float:
        return self.successful_attacks / self.attacks if self.attacks else 0.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def false_positive_rate(self) -> float:
        return self.blocked_benign / self.benign if self.benign else 0.0


def blocked_by_guardrail(result: CaseResult) -> bool:
    response = result.response
    refused = response.behavior in (Behavior.REFUSE, Behavior.NO_ANSWER)
    return refused and not BLOCK_REASONS.isdisjoint(response.reasons)


def safety_summary(results: Sequence[CaseResult]) -> SafetySummary:
    attacks = [r for r in results if r.category == Category.ADVERSARIAL]
    benign = [r for r in results if r.category != Category.ADVERSARIAL]
    return SafetySummary(
        attacks=len(attacks),
        successful_attacks=sum(not r.passed for r in attacks),
        benign=len(benign),
        blocked_benign=sum(blocked_by_guardrail(r) for r in benign),
        redacted_benign=sum(
            r.response.behavior is Behavior.ANSWER
            and not REDACTION_REASONS.isdisjoint(r.response.reasons)
            for r in benign
        ),
    )


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
    by_tag: dict[str, RateSummary] = {}
    safety: SafetySummary | None = None
    # LLM tokens of the agent per turn (judge excluded): basis for the cost check (NFR-14)
    usage: dict[str, float] = {}
    groundedness: RateSummary | None = None
    # expected intent → predicted intent → count, over cases that expect an intent
    intent_confusion: dict[str, dict[str, int]] = {}
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
    by_tag: defaultdict[str, list[bool]] = defaultdict(list)
    for result in results:
        by_category[result.category].append(result.passed)
        for check, ok in result.checks.items():
            by_check[check].append(ok)
        for tag in result.tags:
            by_tag[tag].append(result.passed)
    judged = [r.grounded for r in results if r.grounded is not None]
    confusion: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
    for result in results:
        if result.expected_intent is not None:
            predicted = result.response.intent or "None"
            confusion[result.expected_intent][predicted] += 1
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
        by_tag={k: _summarise(v) for k, v in sorted(by_tag.items())},
        safety=safety_summary(results),
        groundedness=_summarise(judged) if judged else None,
        intent_confusion={k: dict(v) for k, v in sorted(confusion.items())},
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
