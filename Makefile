.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test check eval run up down

help:  ## Show available targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-10s %s\n", $$1, $$2}'

install:  ## Install dependencies and git hooks
	uv sync
	uv run pre-commit install

lint:  ## Lint and check formatting
	uv run ruff check .
	uv run ruff format --check .

format:  ## Auto-fix lint issues and format code
	uv run ruff check --fix .
	uv run ruff format .

typecheck:  ## Static type check (mypy strict)
	uv run mypy

test:  ## Run unit and data tests with coverage
	uv run pytest --cov --cov-report=term-missing -m "not integration and not llm"

check: lint typecheck test  ## Everything CI runs

eval:  ## Run the evaluation dataset against the current agent
	uv run customer-agent-eval --dataset eval/datasets/core.jsonl --day 1

run:  ## Start the API with auto-reload
	uv run uvicorn --factory customer_agent.main:create_app --reload

up:  ## Start local infrastructure (Postgres + pgvector)
	docker compose up -d --wait

down:  ## Stop local infrastructure
	docker compose down
