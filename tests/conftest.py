from datetime import date
from pathlib import Path

import pytest

from customer_agent.evaluation.schema import DATASET_REFERENCE_DATE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = PROJECT_ROOT / "data" / "knowledge"
SEED_DIR = PROJECT_ROOT / "data" / "seed"
CORE_DATASET = PROJECT_ROOT / "eval" / "datasets" / "core.jsonl"

REFERENCE_DATE = DATASET_REFERENCE_DATE
TENANT_ID = "nordelektro"


@pytest.fixture
def reference_date() -> date:
    return REFERENCE_DATE


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--embeddings",
        choices=["hashing", "openai"],
        default="hashing",
        help="Embedding provider for integration tests. 'openai' uses the real model from .env "
        "and also enables tests marked 'llm'.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    real = config.getoption("--embeddings") == "openai"
    skip_llm = pytest.mark.skip(reason="needs the real embedding model: --embeddings=openai")
    for item in items:
        if "tests/integration" in str(item.path):
            item.add_marker(pytest.mark.integration)
        if "llm" in item.keywords and not real:
            item.add_marker(skip_llm)
