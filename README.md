# Mark Anomaly Detection Benchmark

<p align="center">
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"></a>
  <a href="https://huggingface.co/"><img alt="Hugging Face" src="https://img.shields.io/badge/Hugging%20Face-FDEE21?style=for-the-badge&logo=huggingface&logoColor=black"></a>
  <a href="https://docs.astral.sh/uv/"><img alt="uv" src="https://img.shields.io/badge/uv-DE5FE9?style=for-the-badge&logo=uv&logoColor=white"></a>
  <a href="https://docs.astral.sh/ruff/"><img alt="Ruff" src="https://img.shields.io/badge/Ruff-D7FF64?style=for-the-badge&logo=ruff&logoColor=black"></a>
</p>

A reproducible, manifest-driven benchmark for multivariate time-series anomaly detection. It ingests source data locally, refines it into a common schema, and provides the foundations for comparable model scoring and evaluation.

## Reference Benchmarks

This repository uses [mTSBench](https://github.com/PLAN-Lab/mTSBench) and [TSB-AD](https://github.com/TheDatumOrg/TSB-AD) as reference benchmarks for datasets, evaluation patterns, and comparative review. They are not runtime dependencies or implementation authorities; this benchmark owns its manifests, protocols, scoring, and evaluation semantics.

## Setup

```sh
make setup
```

This installs `uv` if necessary and synchronizes the locked Python 3.11 project environment, including development dependencies.

## Usage

Prepare one dataset from its manifest:

```sh
make prepare DATASET=calit2
```

Prepare all versioned dataset manifests:

```sh
make prepare-all
```

Individual stages are also available:

```sh
make ingest DATASET=calit2
make refine DATASET=calit2
make ingest-all
```

Run quality checks:

```sh
make format
make lint
make test
```

## Data Lifecycle

Each dataset begins with a versioned manifest in `data/manifests/<dataset-id>.yaml`. The manifest declares its source revision, split files, native raw schema, refined schema, and SHA-256 checksums.

```text
source repository or archive
-> data/raw/<dataset-id>/
-> data/refined/<dataset-id>/
-> benchmark models and evaluation
```

Raw data preserves source fields. Refinement produces the benchmark schema:

```text
timestamp,feature_0,...,feature_n,label
```

Raw and refined files are local artifacts and are not committed to Git. The manifest records their provenance and checksums.

## Current Architecture

```text
main.py                         executable CLI shim
config/logging.yaml             project-wide logging-level policy
data/manifests/                 versioned dataset provenance and schemas
src/mark_ad_benchmark/
├── cli.py                      command parsing and dispatch
├── logging.py                  pluggable logger and outputs
├── datasets/
│   ├── ingest.py               source download and raw validation
│   ├── loader.py               refined split loading and validation
│   └── refine.py               raw-to-refined conversion
├── protocols/
│   ├── base.py                  Protocol run contract
│   └── inductive.py             train-only fit and held-out split scoring
└── models/
    └── base.py                 AnomalyModel fit/score contract
tests/                          unit tests
```

`main.py` contains no application logic. The package CLI dispatches commands to dataset workflows and, later, experiment execution. Models own anomaly-score calculations; protocols prepare data; evaluation owns metric and threshold policy.

## Development Rules

- Use `uv` through the provided Make targets; do not use global `pip` or direct Python execution.
- Keep `uv.lock` synchronized with `pyproject.toml`.
- Validate raw data before it reaches models.
- Register every module in `config/logging.yaml` and give operational modules a configured component logger; configured module levels must be `info` before committing.
- Do not commit raw/refined data, model weights, or generated run artifacts.
