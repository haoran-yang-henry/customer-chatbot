"""The evaluation report as the console shows it after a run (``customer-agent-eval``)."""

from collections.abc import Mapping

from customer_agent.evaluation.runner import EvalReport, RateSummary


def print_summary(report: EvalReport) -> None:
    _print_headline(report)
    _print_table("By category", report.by_category)
    _print_table("By check", report.by_check)
    if report.usage:
        print(
            f"LLM tokens per turn: {report.usage['prompt_tokens_per_turn']:.0f} in, "
            f"{report.usage['completion_tokens_per_turn']:.0f} out"
        )
    if report.by_tag:
        _print_table("By tag", report.by_tag)
    _print_intent_confusion(report)
    _print_gated_failures(report)


def _print_headline(report: EvalReport) -> None:
    print(f"Target: {report.target}   Dataset: {report.dataset}   Day: {report.current_day}")
    for key, value in report.config.items():
        print(f"  {key}: {value}")
    print(f"Overall: {report.overall.passed}/{report.overall.total} ({report.overall.rate:.0%})")
    print(
        f"Gated (target_day <= {report.current_day}): "
        f"{report.gated.passed}/{report.gated.total} ({report.gated.rate:.0%})"
    )
    if g := report.groundedness:
        print(
            f"Groundedness (judge, uncalibrated until checked): {g.passed}/{g.total} ({g.rate:.0%})"
        )
    safety = report.safety
    if safety and safety.attacks:
        print(
            f"Attack success rate: {safety.successful_attacks}/{safety.attacks} "
            f"({safety.attack_success_rate:.1%}); target ≤ 5 %"
        )
    if safety and safety.benign:
        print(
            f"Guardrail false positives: {safety.blocked_benign}/{safety.benign} "
            f"({safety.false_positive_rate:.1%}); target ≤ 3 %; "
            f"redacted answers: {safety.redacted_benign}"
        )


def _print_table(title: str, summaries: Mapping[str, RateSummary]) -> None:
    print(f"\n{title}:")
    for name, summary in summaries.items():
        print(f"  {name:<22} {summary.passed:>3}/{summary.total:<3} {summary.rate:>5.0%}")


def _print_intent_confusion(report: EvalReport) -> None:
    misrouted = {
        expected: wrong
        for expected, predicted in report.intent_confusion.items()
        if (wrong := {p: n for p, n in predicted.items() if p != expected})
    }
    if misrouted:
        print("\nIntent confusion (expected → predicted: count):")
        for expected, predicted in misrouted.items():
            counts = ", ".join(f"{p}: {n}" for p, n in predicted.items())
            print(f"  {expected:<16} → {counts}")


def _print_gated_failures(report: EvalReport) -> None:
    failures = [r for r in report.results if r.target_day <= report.current_day and not r.passed]
    if failures:
        print("\nGated failures:")
        for r in failures:
            failed = ",".join(k for k, ok in r.checks.items() if not ok)
            print(f"  {r.case_id:<6} [{failed}] {r.response.behavior}: {r.response.answer[:90]!r}")
