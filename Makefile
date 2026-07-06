.PHONY: dev install test test-slow bench lint typecheck clean generate-data

SHELL := /bin/bash

dev: install

install:
	pip install -e ".[dev]"

pre-commit:
	pre-commit install 2>/dev/null || true

test:
	python -m pytest -x -m "not slow and not benchmark" -v

test-slow:
	python -m pytest -x -v -m "slow or benchmark"

bench:
	python -m pytest tests/test_model/test_latency_budget.py -x -v -m "benchmark"

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

typecheck:
	mypy src/ tests/

format:
	ruff format src/ tests/

clean:
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete

generate-data:
	python scripts/generate_mock_data.py
