.PHONY: format lint test run

format:
	uv run ruff format .

lint:
	uv run ruff check .

test:
	uv run pytest

run:
	uv run python main.py $(ARGS)
