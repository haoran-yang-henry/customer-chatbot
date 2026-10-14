"""``customer-agent-eval --dataset eval/datasets/core.jsonl --day 3 --target agent [--judge]``.

``--dataset`` may be repeated and ``--tag`` selects cases, e.g. the CI smoke gate:
``--dataset core.jsonl --dataset redteam.jsonl --tag smoke --min-pass-rate 1``."""

import argparse
import asyncio
import faulthandler
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from customer_agent.evaluation.runner import EvalReport, build_report, run_cases
from customer_agent.evaluation.schema import EvalCase, load_dataset
from customer_agent.evaluation.targets import TARGETS
from customer_agent.observability import configure_logging
from customer_agent.observability.metrics import token_totals


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an evaluation dataset against an agent.")
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--tag", help="Only cases with this tag (e.g. smoke).")
    parser.add_argument("--target", choices=sorted(TARGETS), default="placeholder")
    parser.add_argument("--day", type=int, required=True, help="Current project day (1-8).")
    parser.add_argument("--out", type=Path, help="Result file (default: eval/results/<ts>.json).")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--judge", action="store_true", help="Run the groundedness LLM judge.")
    parser.add_argument(
        "--calibration-out", type=Path, help="Write judged answers for human labelling."
    )
    parser.add_argument(
        "--check-calibration", type=Path, help="Report judge/human agreement for a labelled file."
    )
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        help="Exit with status 1 if the gated pass rate is below this value (0-1).",
    )
    parser.add_argument("--log-level", default="WARNING")
    return parser.parse_args(argv)


def _print_summary(report: EvalReport) -> None:
    print(f"Target: {report.target}   Dataset: {report.dataset}   Day: {report.current_day}")
    for key, value in report.config.items():
        print(f"  {key}: {value}")
    print(f"Overall: {report.overall.passed}/{report.overall.total} ({report.overall.rate:.0%})")
    print(
        f"Gated (target_day <= {report.current_day}): "
        f"{report.gated.passed}/{report.gated.total} ({report.gated.rate:.0%})"
    )
    if report.groundedness:
        g = report.groundedness
        print(
            f"Groundedness (judge, uncalibrated until checked): {g.passed}/{g.total} ({g.rate:.0%})"
        )
    if report.safety and report.safety.attacks:
        sa = report.safety
        print(
            f"Attack success rate: {sa.successful_attacks}/{sa.attacks} "
            f"({sa.attack_success_rate:.1%}); target ≤ 5 %"
        )
    if report.safety and report.safety.benign:
        sa = report.safety
        print(
            f"Guardrail false positives: {sa.blocked_benign}/{sa.benign} "
            f"({sa.false_positive_rate:.1%}); target ≤ 3 %; redacted answers: {sa.redacted_benign}"
        )
    print("\nBy category:")
    for name, summary in report.by_category.items():
        print(f"  {name:<22} {summary.passed:>3}/{summary.total:<3} {summary.rate:>5.0%}")
    print("\nBy check:")
    for name, summary in report.by_check.items():
        print(f"  {name:<22} {summary.passed:>3}/{summary.total:<3} {summary.rate:>5.0%}")
    if report.usage:
        print(
            f"LLM tokens per turn: {report.usage['prompt_tokens_per_turn']:.0f} in, "
            f"{report.usage['completion_tokens_per_turn']:.0f} out"
        )
    if report.by_tag:
        print("\nBy tag:")
        for name, summary in report.by_tag.items():
            print(f"  {name:<22} {summary.passed:>3}/{summary.total:<3} {summary.rate:>5.0%}")
    misrouted = {
        expected: {p: n for p, n in predicted.items() if p != expected}
        for expected, predicted in report.intent_confusion.items()
    }
    misrouted = {k: v for k, v in misrouted.items() if v}
    if misrouted:
        print("\nIntent confusion (expected → predicted: count):")
        for expected, predicted in misrouted.items():
            wrong = ", ".join(f"{p}: {n}" for p, n in predicted.items())
            print(f"  {expected:<16} → {wrong}")
    due = [r for r in report.results if r.target_day <= report.current_day]
    failures = [r for r in due if not r.passed]
    if failures:
        print("\nGated failures:")
        for r in failures:
            failed = ",".join(k for k, ok in r.checks.items() if not ok)
            print(f"  {r.case_id:<6} [{failed}] {r.response.behavior}: {r.response.answer[:90]!r}")


async def _judge(report: EvalReport, cases: list[EvalCase]) -> EvalReport:
    from customer_agent.bootstrap import create_llm_provider
    from customer_agent.config import get_settings
    from customer_agent.evaluation.judge import GroundednessJudge, judgeable
    from customer_agent.infrastructure.knowledge_files import load_knowledge_directory
    from customer_agent.infrastructure.prompts import load_prompt

    settings = get_settings()
    judge = GroundednessJudge(
        create_llm_provider(settings),
        load_prompt("judge", 1),
        load_knowledge_directory(settings.knowledge_dir),
    )
    results = []
    for result in report.results:
        if judgeable(result):
            verdict = await judge.judge(result.response.answer, result.response.sources)
            result = result.model_copy(update={"grounded": verdict.grounded})
        results.append(result)
    config = {**report.config, "judge_prompt": "judge/v1"}
    judged = build_report(
        results,
        target=report.target,
        dataset=report.dataset,
        current_day=report.current_day,
        started_at=report.started_at,
        config=config,
    )
    return judged.model_copy(update={"usage": report.usage})  # measured before judging


async def _run(args: argparse.Namespace, cases: list[EvalCase]) -> EvalReport:
    started_at = datetime.now(UTC)
    before = token_totals()
    async with TARGETS[args.target]() as (agent, config):
        results = await run_cases(cases, agent, current_day=args.day, concurrency=args.concurrency)
    after = token_totals()
    report = build_report(
        results,
        target=args.target,
        dataset=",".join(str(d) for d in args.dataset) + (f" [tag={args.tag}]" if args.tag else ""),
        current_day=args.day,
        started_at=started_at,
        config=config,
    )
    turns = len(results)
    if turns and after["prompt"] > before["prompt"]:
        usage = {
            f"{kind}_tokens_per_turn": round((after[kind] - before[kind]) / turns, 1)
            for kind in ("prompt", "completion")
        }
        report = report.model_copy(update={"usage": usage})
    return await _judge(report, cases) if args.judge else report


def select_cases(datasets: list[Path], tag: str | None) -> list[EvalCase]:
    cases = [case for path in datasets for case in load_dataset(path)]
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("case ids must be unique across the selected datasets")
    selected = [case for case in cases if tag is None or tag in case.tags]
    if not selected:
        raise ValueError(f"no cases with tag {tag!r}")
    return selected


def main(argv: list[str] | None = None) -> int:
    # A native crash (seen once on Day 6, exit 139) then leaves a Python traceback in the log.
    faulthandler.enable()
    args = _parse_args(argv)
    configure_logging(args.log_level)

    if args.check_calibration:
        from customer_agent.evaluation.judge import agreement

        agreeing, labelled = agreement(args.check_calibration)
        rate = agreeing / labelled if labelled else 0.0
        print(
            f"Judge agrees with human labels on {agreeing}/{labelled} ({rate:.0%}); target ≥ 90 %"
        )
        return 0

    cases = select_cases(args.dataset, args.tag)
    report = asyncio.run(_run(args, cases))

    out: Path = args.out or Path("eval/results") / (
        f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{args.target}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")

    if args.calibration_out:
        from customer_agent.evaluation.judge import calibration_rows

        questions = {c.id: c.turns[-1].content for c in cases}
        rows = calibration_rows(report.results, questions)
        args.calibration_out.parent.mkdir(parents=True, exist_ok=True)
        args.calibration_out.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
        )
        print(f"Calibration sheet ({len(rows)} rows) written to {args.calibration_out}")

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
