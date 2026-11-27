.DEFAULT_GOAL := help

.PHONY: help setup sync format lint test train evaluate run visualize publish ingest refine prepare ingest-all prepare-all

UV := $(shell command -v uv 2>/dev/null || printf '%s' "$(HOME)/.local/bin/uv")

help:
	@printf '%s\n' \
		'Available commands:' \
		'  make setup                      Install uv if needed and synchronize dependencies.' \
		'  make sync                       Synchronize dependencies.' \
		'  make format                     Format Python code with Ruff.' \
		'  make lint                       Lint Python code with Ruff.' \
		'  make test                       Run the test suite.' \
		'  make train EXPERIMENT=<id>       Fit and calibrate an experiment.' \
		'  make evaluate EXPERIMENT=<id> VALIDATED_RUN=<id> Test a saved validated artifact.' \
		'  make run EXPERIMENT=<id>         Train then test an experiment.' \
		'  make visualize                    Write an interactive test-run HTML dashboard.' \
		'  make publish                      Package selected visualizations for GitHub Pages.' \
		'  make ingest DATASET=<id>        Download and validate one raw dataset.' \
		'  make refine DATASET=<id>        Validate and refine one local dataset.' \
		'  make prepare DATASET=<id>       Ingest then refine one dataset.' \
		'  make ingest-all                 Ingest every dataset manifest.' \
		'  make prepare-all                Ingest then refine every dataset manifest.'

setup:
	@if ! command -v uv >/dev/null 2>&1 && [ ! -x "$(HOME)/.local/bin/uv" ]; then \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi
	"$(UV)" sync --all-groups --locked
	git config core.hooksPath .githooks

sync:
	"$(UV)" sync --all-groups --locked

format:
	uv run ruff format .

lint:
	uv run ruff check .

test:
	uv run pytest

run:
	PYTHONPATH=src uv run python main.py run $(EXPERIMENT)

train:
	PYTHONPATH=src uv run python main.py train $(EXPERIMENT)

evaluate:
	PYTHONPATH=src uv run python main.py test $(EXPERIMENT) $(VALIDATED_RUN)

visualize:
	PYTHONPATH=src uv run python main.py visualize

publish:
	PYTHONPATH=src uv run python main.py publish

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
