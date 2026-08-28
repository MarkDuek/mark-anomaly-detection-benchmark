# Mark's Multivariate Anomaly Detection Benchmark

<p align="center">
  <a href="https://github.com/MarkDuek/mark-anomaly-detection-benchmark"><img alt="Version 0.1.0" src="https://img.shields.io/badge/version-0.1.0-4f46e5?style=for-the-badge"></a>
  <a href="https://www.python.org/"><img alt="Python 3.11" src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white"></a>
  <a href="https://huggingface.co/"><img alt="Hugging Face" src="https://img.shields.io/badge/Hugging%20Face-FDEE21?style=for-the-badge&logo=huggingface&logoColor=black"></a>
  <a href="https://docs.astral.sh/uv/"><img alt="uv" src="https://img.shields.io/badge/uv-DE5FE9?style=for-the-badge&logo=uv&logoColor=white"></a>
  <a href="https://docs.astral.sh/ruff/"><img alt="Ruff" src="https://img.shields.io/badge/Ruff-D7FF64?style=for-the-badge&logo=ruff&logoColor=black"></a>
</p>

A reproducible, manifest-driven benchmark for multivariate time-series anomaly detection. It owns dataset provenance, protocol lifecycles, anomaly-score generation, threshold calibration, and evaluation.

## Reference Benchmarks

This repository uses [mTSBench](https://github.com/PLAN-Lab/mTSBench) and [TSB-AD](https://github.com/TheDatumOrg/TSB-AD) as reference benchmarks for datasets, evaluation patterns, and comparative review. They are not runtime dependencies or implementation authorities.

## Current Architecture

```text
config/                         logging policy and component catalogs
data/manifests/                 versioned dataset provenance, schemas, and checksums
docs/                           Astro/Starlight documentation site
experiments/                    versioned dataset, model, protocol, and metric definitions
src/mark_ad_benchmark/
├── cli.py                      command dispatch
├── experiment.py               train, validation, and test orchestration
├── datasets/                   Hugging Face ingestion, refinement, and validated loading
├── metrics/                    point, event, ranking, affiliation, VUS, and assembly metrics
├── models/                     inductive PCA and transductive Robust PCA
├── protocols/                  inductive and transductive data-use policies
├── thresholds/                 validation-score threshold selection
└── visualization/              interactive saved-run dashboard
tests/                          deterministic unit and integration tests
```

## Requirements

- Python `>=3.11,<3.12`
- `uv` for the benchmark environment; `make setup` installs it when needed
- Node.js and npm only for the documentation site

## Setup

```sh
make setup
```

## Documentation Site

Install the documentation dependencies and start the local Astro server:

```sh
cd docs
npm ci
npm run dev
```

Build the static site:

```sh
cd docs
npm run build
```

## Quick Start

```sh
make prepare DATASET=calit2
make run EXPERIMENT=calit2-pca-inductive
make visualize
```

Detailed benchmark and documentation-site workflows are in the [Usage guide](docs/src/content/docs/usage.mdx).

## Data Lifecycle

Each manifest records a pinned Hugging Face revision, split paths, raw and refined schemas, and SHA-256 checksums.

```text
Hugging Face dataset revision
-> data/raw/<dataset-id>/
-> data/refined/<dataset-id>/
-> protocols, models, thresholds, and metrics
```

Raw and refined files are local artifacts and are not committed to Git.

## Development

```sh
make format
make lint
make test
```

Do not commit raw or refined data, model artifacts, or generated run artifacts.
