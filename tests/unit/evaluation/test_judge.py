import json
from pathlib import Path

import pytest

from customer_agent.evaluation.judge import agreement, parse_verdict


@pytest.mark.parametrize(
    ("text", "grounded"),
    [
        ('{"grounded": true, "unsupported_claims": []}', True),
        ('Here you go:\n{"grounded": false, "unsupported_claims": ["60 Tage"]}', False),
        ("not json at all", None),
        ('{"grounded": "yes"}', None),
    ],
)
def test_parse_verdict_is_tolerant_but_strict_about_types(text: str, grounded: bool | None) -> None:
    assert parse_verdict(text).grounded is grounded


def test_agreement_counts_only_labelled_rows(tmp_path: Path) -> None:
    rows = [
        {"judge_grounded": True, "human_grounded": True},
        {"judge_grounded": True, "human_grounded": False},
        {"judge_grounded": False, "human_grounded": None},
    ]
    path = tmp_path / "labels.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    assert agreement(path) == (1, 2)
