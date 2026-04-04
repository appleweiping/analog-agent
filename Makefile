PYTHON ?= python

.PHONY: install format lint test run-api

install:
	$(PYTHON) -m pip install -e .[dev]

format:
	$(PYTHON) -m ruff format .

lint:
	$(PYTHON) -m ruff check .

test:
	$(PYTHON) -m pytest

run-api:
	$(PYTHON) -m uvicorn apps.api_server.main:app --host 0.0.0.0 --port 8000 --reload
