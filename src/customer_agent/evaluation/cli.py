"""Command-line entry point: ``customer-agent-eval --dataset eval/datasets/core.jsonl --day 1``."""

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

from customer_agent.evaluation.runner import EvalReport, run_evaluation
from customer_agent.evaluation.schema import load_dataset
from customer_agent.evaluation.targets import TARGETS


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an evaluation dataset against an agent.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--target", choices=sorted(TARGETS), default="placeholder")
    parser.add_argument("--day", type=int, required=True, help="Current project day (1-8).")
    parser.add_argument("--out", type=Path, help="Result file (default: eval/results/<ts>.json).")
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        help="Exit with status 1 if the gated pass rate is below this value (0-1).",
    )
    return parser.parse_args(argv)


def _print_summary(report: EvalReport) -> None:
    print(f"Target: {report.target}   Dataset: {report.dataset}   Day: {report.current_day}")
    print(f"Overall: {report.overall.passed}/{report.overall.total} ({report.overall.rate:.0%})")
    print(
        f"Gated (target_day <= {report.current_day}): "
        f"{report.gated.passed}/{report.gated.total} ({report.gated.rate:.0%})"
    )
    print("\nBy category:")
    for name, summary in report.by_category.items():
        print(f"  {name:<22} {summary.passed:>3}/{summary.total:<3} {summary.rate:>5.0%}")
    print("\nBy check:")
    for name, summary in report.by_check.items():
        print(f"  {name:<22} {summary.passed:>3}/{summary.total:<3} {summary.rate:>5.0%}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cases = load_dataset(args.dataset)
    agent = TARGETS[args.target]()
    report = asyncio.run(
        run_evaluation(
            cases, agent, target=args.target, dataset=str(args.dataset), current_day=args.day
        )
    )

    out: Path = args.out or Path("eval/results") / (
        f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{args.target}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    _print_summary(report)
    print(f"\nWritten to {out}")

    if (
        args.min_pass_rate is not None
        and report.gated.total
        and (report.gated.rate < args.min_pass_rate)
    ):
        print(f"FAILED: gated pass rate below {args.min_pass_rate:.0%}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
