.PHONY: help install lint format test generate run clean

PYTHON ?= python

help:
	@echo "Causal-Retain Engine: Developer Commands"
	@echo "----------------------------------------"
	@echo "make install   - Install dependencies"
	@echo "make lint      - Check code style and static analysis"
	@echo "make format    - Auto-format code using ruff"
	@echo "make test      - Execute unit test suite"
	@echo "make generate  - Generate synthetic telemetry dataset"
	@echo "make run       - Launch Streamlit dashboard"
	@echo "make clean     - Clean cache and temporary files"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

lint:
	ruff check .

format:
	ruff format .

test:
	pytest tests/

generate:
	$(PYTHON) data/generate_telemetry.py

run:
	streamlit run app/streamlit_app.py

clean:
	@powershell -Command "Get-ChildItem -Path . -Include __pycache__, .pytest_cache, .ruff_cache -Recurse -Directory -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force"
