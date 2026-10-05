import asyncio
from pathlib import Path

from customer_agent.evaluation.cli import main
from customer_agent.evaluation.runner import run_evaluation
from customer_agent.evaluation.schema import load_dataset
from customer_agent.evaluation.targets import PlaceholderAgent
from tests.conftest import CORE_DATASET


def test_gated_summary_only_counts_cases_due_by_current_day() -> None:
    cases = load_dataset(CORE_DATASET)

    report = asyncio.run(
        run_evaluation(
            cases, PlaceholderAgent(), target="placeholder", dataset="core", current_day=3
        )
    )

    assert report.overall.total == len(cases)
    assert report.gated.total == sum(1 for case in cases if case.target_day <= 3)


def test_cli_writes_report_and_enforces_min_pass_rate(tmp_path: Path) -> None:
    out = tmp_path / "result.json"
    args = ["--dataset", str(CORE_DATASET), "--day", "3", "--out", str(out)]

    assert main(args) == 0
    assert out.exists()
    assert main([*args, "--min-pass-rate", "0.9"]) == 1
