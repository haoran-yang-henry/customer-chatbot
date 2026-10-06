"""Loads versioned prompt templates from ``prompts/<name>/v<version>.yaml``.

The ``ref`` (``name/vN``) of every prompt used is recorded in evaluation results.
"""

from pathlib import Path

import yaml

from customer_agent.application.prompts import PromptTemplate

PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"


def load_prompt(name: str, version: int, directory: Path = PROMPTS_DIR) -> PromptTemplate:
    path = directory / name / f"v{version}.yaml"
    prompt = PromptTemplate.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    if (prompt.name, prompt.version) != (name, version):
        raise ValueError(f"{path}: declares {prompt.ref}, expected {name}/v{version}")
    return prompt
