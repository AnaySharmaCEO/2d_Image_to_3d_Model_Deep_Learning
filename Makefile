.PHONY: help install install-cuda ui test lint format clean

help:
	@echo "Available commands:"
	@echo "  make install       - Install base dependencies"
	@echo "  make install-cuda  - Install base + PyTorch / CUDA acceleration packages"
	@echo "  make ui            - Launch the Gradio Web UI"
	@echo "  make test          - Run pytest test suite"
	@echo "  make lint          - Run ruff code linter"
	@echo "  make format        - Format code using black and ruff"
	@echo "  make clean         - Remove temporary cache and build files"

install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

install-cuda:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt
	python -m pip install -e .[cuda]

ui:
	python app.py

test:
	pytest tests/

lint:
	ruff check src/ tests/

format:
	black src/ tests/
	ruff check --fix src/ tests/

clean:
	python -c "import shutil, glob, os; [shutil.rmtree(p, ignore_errors=True) for p in glob.glob('**/__pycache__', recursive=True) + glob.glob('*.egg-info') + glob.glob('.pytest_cache')]"
