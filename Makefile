.PHONY: help install lint format test data run-api clean

help:
	@echo "Available commands:"
	@echo "  make install  - Install all dependencies via uv sync"
	@echo "  make lint     - Run code quality checks (ruff & mypy)"
	@echo "  make format   - Automatically format code (ruff format & fix)"
	@echo "  make test     - Run tests with coverage"
	@echo "  make data     - Run data pipeline / initialize datasets"
	@echo "  make run-api  - Launch FastAPI dev server with reload"
	@echo "  make clean    - Remove build and cache artifacts"

install:
	uv sync --all-extras

lint:
	uv run ruff check .
	uv run mypy src

format:
	uv run ruff format .
	uv run ruff check --fix .

test:
	uv run pytest

data:
	uv run python -m credit_policy_optimizer.data

run-api:
	uv run uvicorn credit_policy_optimizer.api:app --reload --host 0.0.0.0 --port 8000

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov dist build *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
