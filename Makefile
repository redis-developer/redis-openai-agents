.PHONY: help install test lint format check clean examples docs docs-serve docs-clean

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies with uv
	uv sync --all-extras

test:  ## Run tests
	uv run pytest -v

test-all:  ## Run all tests including API tests
	uv run pytest --run-api-tests -v

test-cov:  ## Run tests with coverage
	uv run pytest --cov=src --cov-report=html --cov-report=term

lint:  ## Run linter
	uv run ruff check src/ tests/

format:  ## Format code
	uv run ruff format src/ tests/

format-check:  ## Check code formatting
	uv run ruff format --check src/ tests/

mypy:  ## Run type checker
	uv run mypy src/

check: format-check lint mypy test  ## Run all checks

clean:  ## Remove build artifacts and cache
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

examples:  ## Start Jupyter notebook examples
	cd examples && docker compose up

examples-build:  ## Rebuild and start examples
	cd examples && docker compose up --build

examples-down:  ## Stop example environment
	cd examples && docker compose down

examples-logs:  ## View example logs
	cd examples && docker compose logs -f

docs:  ## Build documentation
	uv run sphinx-build -b html docs docs/_build/html

docs-serve:  ## Serve documentation locally
	uv run python -m http.server 8000 --directory docs/_build/html

docs-clean:  ## Clean documentation build
	rm -rf docs/_build

.DEFAULT_GOAL := help
