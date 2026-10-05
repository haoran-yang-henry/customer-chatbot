from typing import Any

import pytest

from customer_agent.domain.conversation import AgentResponse, Behavior
from customer_agent.domain.intents import Intent
from customer_agent.evaluation.schema import EvalCase
from customer_agent.evaluation.scoring import (
    contains_all_groups,
    contains_none,
    score_case,
    source_recall,
)


def make_case(**overrides: Any) -> EvalCase:
    data: dict[str, Any] = {
        "id": "N-001",
        "category": "normal",
        "target_day": 3,
        "customer_id": None,
        "turns": [{"role": "user", "content": "Wie lange kann ich zurückgeben?"}],
        "expected_intent": "RETURN_POLICY",
        "expected_behavior": "answer",
        "expected_sources": ["policy-returns-v5"],
        "expected_tool": None,
        "must_contain": [["30 Tage", "30 Tagen"]],
        "must_not_contain": ["5,95"],
    }
    data.update(overrides)
    return EvalCase.model_validate(data)


GOOD = AgentResponse(
    answer="Sie haben 30 Tage Zeit. Die Rücksendung ist kostenlos.",
    behavior=Behavior.ANSWER,
    intent=Intent.RETURN_POLICY,
    sources=["policy-returns-v5"],
)


class TestPhraseChecks:
    def test_group_passes_if_any_alternative_matches_case_insensitively(self) -> None:
        assert contains_all_groups("Innerhalb von 30 TAGEN", [["30 Tage", "30 Tagen"]])

    def test_every_group_must_match(self) -> None:
        assert not contains_all_groups("30 Tage", [["30 Tage"], ["kostenlos"]])

    def test_contains_none_detects_forbidden_phrase(self) -> None:
        assert not contains_none("Kosten: 5,95 €", ["5,95"])


class TestSourceRecall:
    def test_none_when_no_sources_expected(self) -> None:
        assert source_recall([], ["a"]) is None

    def test_fraction_of_expected_sources_cited(self) -> None:
        assert source_recall(["a", "b"], ["b", "c"]) == pytest.approx(0.5)


class TestScoreCase:
    def test_correct_response_passes_all_checks(self) -> None:
        result = score_case(make_case(), GOOD)

        assert result.passed
        assert set(result.checks) == {
            "behavior",
            "intent",
            "sources",
            "must_contain",
            "must_not_contain",
        }

    def test_forbidden_phrase_fails_case(self) -> None:
        response = GOOD.model_copy(update={"answer": "30 Tage, Rücksendung kostet 5,95 €."})

        result = score_case(make_case(), response)

        assert not result.passed
        assert result.checks["must_not_contain"] is False

    def test_unspecified_expectations_are_not_checked(self) -> None:
        case = make_case(
            expected_intent=None, expected_sources=[], must_contain=[], must_not_contain=[]
        )

        assert set(score_case(case, GOOD).checks) == {"behavior"}

    def test_tool_check_requires_expected_tool_in_calls(self) -> None:
        case = make_case(expected_tool="get_order_status")

        assert score_case(case, GOOD).checks["tool"] is False


class TestEvalCaseValidation:
    def test_turns_must_alternate_starting_with_user(self) -> None:
        with pytest.raises(ValueError, match="turn 1"):
            make_case(turns=[{"role": "user", "content": "a"}, {"role": "user", "content": "b"}])

    def test_last_turn_must_be_user(self) -> None:
        with pytest.raises(ValueError, match="last turn"):
            make_case(
                turns=[{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
            )
