ifeq ($(OS),Windows_NT)
PYTHON ?= py -3.12
VENV_PYTHON ?= .venv\Scripts\python.exe
BOOTSTRAP_COMMAND ?= py -3.12 -m venv .venv
else
PYTHON ?= python3
VENV_PYTHON ?= .venv/bin/python
BOOTSTRAP_COMMAND ?= python3 -m venv .venv
endif

.PHONY: install install-dev bootstrap-dev format lint test test-all test-api run-api

install:
	$(PYTHON) -m pip install -e .[dev]

install-dev:
	$(VENV_PYTHON) -m pip install -U pip
	$(VENV_PYTHON) -m pip install -e .[dev]

bootstrap-dev:
	$(BOOTSTRAP_COMMAND)
	$(MAKE) install-dev

format:
	$(PYTHON) -m ruff format .

lint:
	$(PYTHON) -m ruff check .

test:
	$(PYTHON) scripts/run_test_suite.py

test-all:
	$(VENV_PYTHON) scripts/run_test_suite.py --require-api-deps

test-api:
	$(VENV_PYTHON) -m unittest tests.integration.test_interaction_api tests.integration.test_tasking_api

run-api:
	$(PYTHON) -m uvicorn apps.api_server.main:app --host 0.0.0.0 --port 8000 --reload
