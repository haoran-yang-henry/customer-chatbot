from datetime import date
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = PROJECT_ROOT / "data" / "knowledge"
SEED_DIR = PROJECT_ROOT / "data" / "seed"
CORE_DATASET = PROJECT_ROOT / "eval" / "datasets" / "core.jsonl"

# Evaluation reference date; see data/seed/README.md.
REFERENCE_DATE = date(2026, 10, 5)
TENANT_ID = "nordelektro"


@pytest.fixture
def reference_date() -> date:
    return REFERENCE_DATE
