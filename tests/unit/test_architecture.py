"""Enforces the dependency rule of ADR-0001: inner layers never import outer layers."""

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "customer_agent"

FORBIDDEN = {
    "domain": (
        "customer_agent.application",
        "customer_agent.infrastructure",
        "customer_agent.api",
        "langchain",
        "langgraph",
        "langfuse",
    ),
    "application": (
        "customer_agent.infrastructure",
        "customer_agent.api",
        "customer_agent.bootstrap",
        "langchain_openai",
        "langfuse",
        "openai",
        "psycopg",
        "sqlalchemy",
    ),
}


def _imports(path: Path) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


@pytest.mark.parametrize("layer", sorted(FORBIDDEN))
def test_layer_does_not_import_outer_layers(layer: str) -> None:
    violations = [
        f"{path.relative_to(PACKAGE)} imports {module}"
        for path in sorted((PACKAGE / layer).rglob("*.py"))
        for module in _imports(path)
        if module.startswith(FORBIDDEN[layer])
    ]

    assert not violations, "\n".join(violations)


def _create_agent_calls(path: Path) -> list[ast.Call]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "create_agent"
    ]


def test_agents_get_no_tools() -> None:
    """Invariant I1 of the rebuild: no model chooses a tool; ``decide()`` does (ADR-0014)."""
    calls = {
        path.relative_to(PACKAGE): _create_agent_calls(path)
        for path in sorted(PACKAGE.rglob("*.py"))
    }
    agents = {path: found for path, found in calls.items() if found}
    with_tools = [
        f"{path}:{call.lineno}"
        for path, found in agents.items()
        for call in found
        if not any(
            k.arg == "tools" and isinstance(k.value, ast.Constant) and k.value.value is None
            for k in call.keywords
        )
    ]

    assert agents, "no create_agent call found: the check would pass vacuously"
    assert not with_tools, "create_agent without tools=None:\n" + "\n".join(with_tools)
