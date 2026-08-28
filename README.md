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

## Architecture

```text
config/                         # runtime logging configuration
data/
├── manifests/                  # dataset contracts and provenance records
├── raw/                        # source files validated after acquisition
└── refined/                    # normalized benchmark splits ready for use
docs/                           # documentation source and static-site tooling
experiments/                    # reproducible experiment specifications
src/mark_ad_benchmark/
├── cli.py                      # command dispatch
├── experiment.py               # train, validation, and test orchestration
├── datasets/                   # data acquisition, refinement, and schema validation
├── metrics/                    # benchmark metric calculation and assembly
├── models/                     # anomaly-score model implementations
├── protocols/                  # data-access policies for model execution
├── thresholds/                 # validation-based score-to-label calibration
└── visualization/              # saved-run exports for the documentation explorer
tests/                          # automated unit and integration coverage
```

## Requirements

- Python `>=3.11,<3.12`
- `uv` for the benchmark environment; `make setup` installs it when needed
- Node.js and npm only for the documentation site

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

### Setup

```sh
make setup
```

### Download and Prepare a Dataset

```sh
make prepare DATASET=calit2
```

### Run an Experiment

```sh
make run EXPERIMENT=calit2-pca-inductive
make visualize
```

### Visualize a Saved Run

```sh
make visualize
```

Detailed benchmark and documentation-site workflows are in the [Usage guide](docs/src/content/docs/usage.mdx).

## Development

- Use uv through the provided Make targets; do not use global pip or direct Python execution.
- Keep uv.lock synchronized with pyproject.toml.
- Validate raw data before it reaches models.
- Register every module in config/logging.yaml and give operational modules a configured component logger; configured module levels must be info before committing.
- Do not commit raw/refined data, model weights, or generated run artifacts.
- Always run

    ```sh
    make format
    make lint
    make test
    ```

    before committing.
