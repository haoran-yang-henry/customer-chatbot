"""Evaluation dataset schema and loader."""

import json
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from customer_agent.domain.conversation import Behavior, Conversation, Message, Role
from customer_agent.domain.intents import Intent

# All date-dependent expectations in the datasets assume this "today" (data/seed/README.md).
DATASET_REFERENCE_DATE = date(2026, 10, 5)


class Category(StrEnum):
    NORMAL = "normal"
    LANGUAGE = "language"
    AMBIGUOUS = "ambiguous"
    MULTI_TURN = "multi_turn"
    TRANSACTIONAL = "transactional"
    OFF_TOPIC = "off_topic"
    MISSING_KNOWLEDGE = "missing_knowledge"
    KNOWLEDGE_GOVERNANCE = "knowledge_governance"
    ADVERSARIAL = "adversarial"


class EvalCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^[A-Z]-\d{3}$")
    category: Category
    target_day: int = Field(ge=1, le=8)
    customer_id: str | None
    turns: list[Message] = Field(min_length=1)
    expected_intent: Intent | None
    expected_behavior: Behavior
    expected_sources: list[str]
    expected_tool: str | None
    must_contain: list[list[str]] = Field(
        description="Each group passes if any of its phrases occurs (case-insensitive)."
    )
    must_not_contain: list[str]
    notes: str = ""
    # Free-form labels for slicing results, e.g. the attack technique in redteam.jsonl.
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_turns(self) -> Self:
        for index, turn in enumerate(self.turns):
            expected = Role.USER if index % 2 == 0 else Role.ASSISTANT
            if turn.role is not expected:
                raise ValueError(f"turn {index} must be '{expected}', got '{turn.role}'")
        if self.turns[-1].role is not Role.USER:
            raise ValueError("the last turn must come from the user")
        if any(not group for group in self.must_contain):
            raise ValueError("must_contain groups must not be empty")
        return self

    def to_conversation(self) -> Conversation:
        return Conversation(messages=self.turns, customer_id=self.customer_id)


def load_dataset(path: Path) -> list[EvalCase]:
    cases = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                cases.append(EvalCase.model_validate(json.loads(line)))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    ids = [case.id for case in cases]
    duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    if duplicates:
        raise ValueError(f"{path}: duplicate case ids {duplicates}")
    return cases
