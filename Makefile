# ---------------------------------------------------------------------------
# Developer entry points. Every target maps to a command that is also
# documented in the README, so `make` is a convenience rather than a second
# source of truth.
# ---------------------------------------------------------------------------

PYTHON ?= python
PIP ?= $(PYTHON) -m pip
UVICORN ?= $(PYTHON) -m uvicorn
STREAMLIT ?= $(PYTHON) -m streamlit
PYTEST ?= $(PYTHON) -m pytest
RUFF ?= $(PYTHON) -m ruff

.PHONY: help install data train evaluate test lint format typecheck api app docker-build docker-up docker-down clean

help:
	@echo "install      Install the package with dev, API, app and training extras"
	@echo "data         Download the ISOT corpus and build leakage-safe splits"
	@echo "train        Train, compare, calibrate and select the final model"
	@echo "evaluate     Evaluate the saved model on the untouched test split"
	@echo "test         Run the test suite"
	@echo "lint         Run the linter and the formatting check"
	@echo "format       Apply Ruff formatting"
	@echo "typecheck    Run mypy on the package"
	@echo "api          Start the FastAPI service on 127.0.0.1:8000"
	@echo "app          Start the Streamlit UI (requires the API)"
	@echo "docker-up    Build and start the API and UI with Docker Compose"
	@echo "docker-down  Stop the Compose stack"
	@echo "clean        Remove caches and generated reports (keeps data/artifacts)"

install:
	$(PIP) install -e ".[dev,api,app,train]"

data:
	$(PYTHON) scripts/download_data.py
	$(PYTHON) scripts/prepare_data.py

train:
	$(PYTHON) scripts/train.py

evaluate:
	$(PYTHON) scripts/evaluate.py

test:
	$(PYTEST)

lint:
	$(RUFF) check .
	$(RUFF) format --check .

format:
	$(RUFF) format .
	$(RUFF) check --fix .

typecheck:
	$(PYTHON) -m mypy src

api:
	$(UVICORN) fake_news_detection.api.main:app --host 127.0.0.1 --port 8000 --reload

app:
	$(STREAMLIT) run app/streamlit_app.py

docker-build:
	docker compose build

docker-up:
	docker compose up --build

docker-down:
	docker compose down

clean:
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache', '.ruff_cache', '.mypy_cache', 'htmlcov']]"
	$(PYTHON) -c "import pathlib; [p.unlink(missing_ok=True) for p in pathlib.Path('.').glob('*.pyc')]"
