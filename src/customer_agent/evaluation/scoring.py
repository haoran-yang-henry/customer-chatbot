"""Deterministic checks for one evaluation case. Pure functions, no I/O."""

from pydantic import BaseModel, ConfigDict, computed_field

from customer_agent.domain.conversation import AgentResponse
from customer_agent.evaluation.schema import EvalCase


class CaseResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    category: str
    target_day: int
    checks: dict[str, bool]
    source_recall: float | None
    expected_intent: str | None = None
    response: AgentResponse
    grounded: bool | None = None  # LLM judge verdict, only when the judge ran

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        return all(self.checks.values())


def contains_all_groups(text: str, groups: list[list[str]]) -> bool:
    folded = text.casefold()
    return all(any(phrase.casefold() in folded for phrase in group) for group in groups)


def contains_none(text: str, phrases: list[str]) -> bool:
    folded = text.casefold()
    return not any(phrase.casefold() in folded for phrase in phrases)


def source_recall(expected: list[str], cited: list[str]) -> float | None:
    """Share of expected sources that were cited; ``None`` when no sources are expected."""
    if not expected:
        return None
    return len(set(expected) & set(cited)) / len(set(expected))


# Capabilities arrive day by day; a check counts only once its capability exists, so an early
# agent is not failed for missing intents (Day 4) or tools (Day 5).
CHECK_INTRODUCED_ON_DAY = {"intent": 4, "tool": 5}


def score_case(case: EvalCase, response: AgentResponse, current_day: int = 8) -> CaseResult:
    """Apply every applicable check; expectations left unspecified are not checked."""
    checks: dict[str, bool] = {"behavior": response.behavior is case.expected_behavior}
    if case.expected_intent is not None and current_day >= CHECK_INTRODUCED_ON_DAY["intent"]:
        checks["intent"] = response.intent is case.expected_intent
    recall = source_recall(case.expected_sources, response.sources)
    if recall is not None:
        checks["sources"] = recall > 0
    if case.expected_tool is not None and current_day >= CHECK_INTRODUCED_ON_DAY["tool"]:
        checks["tool"] = case.expected_tool in response.tool_calls
    if case.must_contain:
        checks["must_contain"] = contains_all_groups(response.answer, case.must_contain)
    if case.must_not_contain:
        checks["must_not_contain"] = contains_none(response.answer, case.must_not_contain)
    return CaseResult(
        case_id=case.id,
        category=case.category,
        target_day=case.target_day,
        checks=checks,
        source_recall=recall,
        expected_intent=case.expected_intent,
        response=response,
    )
