PY ?= python

.PHONY: install data run fast test lint clean

install:
	$(PY) -m pip install -e ".[dev]"

data:
	$(PY) scripts/download_data.py

run: data
	$(PY) -m secom_spc.pipeline

fast: data
	$(PY) -m secom_spc.pipeline --fast

test:
	$(PY) -m pytest -q

lint:
	ruff check src tests scripts
