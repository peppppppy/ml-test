PYTHON ?= python
POETRY ?= $(PYTHON) -m poetry

.PHONY: install install-dev train predict tune dashboard lint type-check test

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install poetry
	$(POETRY) install

install-dev: install

train:
	$(POETRY) run python -m src.main train

predict:
	$(POETRY) run python -m src.main predict

tune:
	$(POETRY) run python -m src.main tune

dashboard:
	$(POETRY) run streamlit run src/streamlit_app.py

lint:
	$(POETRY) run ruff check src tests

type-check:
	$(POETRY) run mypy src

test:
	$(POETRY) run pytest -q
