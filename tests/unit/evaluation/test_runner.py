import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from customer_agent.domain.conversation import AgentResponse, Behavior
from customer_agent.evaluation.cli import main, select_cases
from customer_agent.evaluation.runner import EvalReport, build_report, run_cases, safety_summary
from customer_agent.evaluation.schema import EvalCase, load_dataset
from customer_agent.evaluation.scoring import CaseResult
from customer_agent.evaluation.targets import PlaceholderAgent
from tests.conftest import CORE_DATASET


def evaluate(cases: list[EvalCase], *, current_day: int) -> EvalReport:
    """What the CLI does without Langfuse: run the cases, build the report."""
    started_at = datetime.now(UTC)
    results = asyncio.run(run_cases(cases, PlaceholderAgent(), current_day=current_day))
    return build_report(
        results,
        target="placeholder",
        dataset="core",
        current_day=current_day,
        started_at=started_at,
    )


def test_gated_summary_only_counts_cases_due_by_current_day() -> None:
    cases = load_dataset(CORE_DATASET)

    report = evaluate(cases, current_day=3)

    assert report.overall.total == len(cases)
    assert report.gated.total == sum(1 for case in cases if case.target_day <= 3)


def test_cli_writes_report_and_enforces_min_pass_rate(tmp_path: Path) -> None:
    # placeholder target: no network, no database
    out = tmp_path / "result.json"
    args = ["--dataset", str(CORE_DATASET), "--day", "3", "--out", str(out)]

    assert main(args) == 0
    assert out.exists()
    assert main([*args, "--min-pass-rate", "0.9"]) == 1


def test_cases_are_selected_by_tag_across_datasets() -> None:
    redteam = CORE_DATASET.parent / "redteam.jsonl"
    smoke = select_cases([CORE_DATASET, redteam], "smoke")

    assert {case.id[0] for case in smoke} > {"X"}  # core and red-team cases
    assert all("smoke" in case.tags for case in smoke)
    assert len(select_cases([CORE_DATASET], None)) == len(load_dataset(CORE_DATASET))
    with pytest.raises(ValueError, match="no cases"):
        select_cases([CORE_DATASET], "no-such-tag")
    with pytest.raises(ValueError, match="unique"):
        select_cases([CORE_DATASET, CORE_DATASET], None)


def result(category: str, passed: bool, **response: object) -> CaseResult:
    return CaseResult(
        case_id="X-001",
        category=category,
        target_day=6,
        checks={"behavior": passed},
        source_recall=None,
        tags=["direct_injection"] if category == "adversarial" else [],
        response=AgentResponse.model_validate({"answer": "a", "behavior": "answer", **response}),
    )


def test_safety_summary_counts_attacks_and_guardrail_false_positives() -> None:
    results = [
        result("adversarial", passed=True, behavior=Behavior.REFUSE, reasons=["prompt_injection"]),
        result("adversarial", passed=False),  # the attack got through
        # benign, blocked by a guardrail → false positive
        result("normal", passed=False, behavior=Behavior.REFUSE, reasons=["prompt_injection_llm"]),
        # benign, refused by design (out of scope) → not a guardrail false positive
        result("off_topic", passed=True, behavior=Behavior.REFUSE),
        # benign, answered with a sentence removed → redacted, not blocked
        result("normal", passed=True, reasons=["unverified_content"]),
    ]

    safety = safety_summary(results)

    assert (safety.attacks, safety.successful_attacks) == (2, 1)
    assert (safety.benign, safety.blocked_benign, safety.redacted_benign) == (3, 1, 1)
    assert safety.attack_success_rate == 0.5
    assert safety.false_positive_rate == 1 / 3


def test_results_are_also_summarised_per_tag() -> None:
    from datetime import UTC, datetime

    report = build_report(
        [result("adversarial", passed=True), result("adversarial", passed=False)],
        target="t",
        dataset="d",
        current_day=6,
        started_at=datetime.now(UTC),
    )

    assert report.by_tag["direct_injection"].passed == 1
    assert report.by_tag["direct_injection"].total == 2


def test_judging_keeps_the_token_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    """The judge rebuilds the report; the agent's token usage was measured before it."""
    import customer_agent.bootstrap
    from customer_agent.evaluation.cli import _judge
    from tests.fakes import FakeLLM

    monkeypatch.setattr(customer_agent.bootstrap, "create_chat_model", lambda _s: FakeLLM())
    cases = load_dataset(CORE_DATASET)
    report = evaluate(cases, current_day=7).model_copy(
        update={"usage": {"prompt_tokens_per_turn": 1.0}}
    )

    judged = asyncio.run(_judge(report, cases))

    assert judged.usage == {"prompt_tokens_per_turn": 1.0}
