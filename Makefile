.PHONY: setup format lint test run ingest refine prepare ingest-all prepare-all

UV := $(shell command -v uv 2>/dev/null || printf '%s' "$(HOME)/.local/bin/uv")

setup:
	@if ! command -v uv >/dev/null 2>&1 && [ ! -x "$(HOME)/.local/bin/uv" ]; then \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi
	"$(UV)" sync --all-groups --locked

format:
	uv run ruff format .

lint:
	uv run ruff check .

test:
	uv run pytest

run:
	PYTHONPATH=src uv run python main.py $(ARGS)

ingest:
	PYTHONPATH=src uv run python main.py ingest $(DATASET)

refine:
	PYTHONPATH=src uv run python main.py refine $(DATASET)

prepare:
	$(MAKE) ingest DATASET=$(DATASET)
	$(MAKE) refine DATASET=$(DATASET)

ingest-all:
	@set -e; for manifest in data/manifests/*.yaml; do \
		dataset=$${manifest##*/}; \
		$(MAKE) ingest DATASET=$${dataset%.yaml}; \
	done

prepare-all:
	@set -e; for manifest in data/manifests/*.yaml; do \
		dataset=$${manifest##*/}; \
		$(MAKE) prepare DATASET=$${dataset%.yaml}; \
	done
