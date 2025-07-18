.PHONY: help install test lint format type-check clean setup dev-setup

help:
	@echo "Available commands:"
	@echo "  install     Install dependencies"
	@echo "  test        Run tests"
	@echo "  lint        Run linting (flake8)"
	@echo "  format      Format code (black, isort)"
	@echo "  type-check  Run type checking (mypy)"
	@echo "  clean       Clean cache and build files"
	@echo "  setup       Setup development environment"
	@echo "  dev-setup   Setup development environment (alias for setup)"

install:
	@if command -v poetry >/dev/null 2>&1; then \
		echo "Using Poetry..."; \
		poetry install; \
	else \
		echo "Using pip..."; \
		pip install -r requirements.txt; \
		pip install -r requirements-dev.txt; \
	fi

test:
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run pytest; \
	else \
		pytest; \
	fi

lint:
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run flake8 .; \
	else \
		flake8 .; \
	fi

format:
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run black .; \
		poetry run isort .; \
	else \
		black .; \
		isort .; \
	fi

type-check:
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run mypy .; \
	else \
		mypy .; \
	fi

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type d -name ".mypy_cache" -delete
	find . -type d -name "*.egg-info" -delete
	find . -type d -name "build" -delete
	find . -type d -name "dist" -delete

setup: install
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run pre-commit install; \
	else \
		pre-commit install; \
	fi
	@echo "Development environment setup complete!"

dev-setup: setup

ci: lint type-check test
	@echo "CI checks completed successfully!"
