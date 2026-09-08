.PHONY: install build-image test lint format setup clean hooks-install hooks-run

install:
	pip install -e ".[dev]"

build-image:
	docker build -t nerdit-runtime:0.1 docker/

test:
	pytest tests/ -m "not hardware" -v

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .

setup: install build-image hooks-install
	@echo "Setup complete. Pre-commit hooks installed. Run 'nerdit init' to start."

hooks-install:
	pre-commit install
	pre-commit install --hook-type pre-push

hooks-run:
	pre-commit run --all-files

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
