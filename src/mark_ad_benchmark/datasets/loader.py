import csv
from dataclasses import dataclass
from pathlib import Path

import yaml

from mark_ad_benchmark.datasets.ingest import sha256, validate_csv
from mark_ad_benchmark.logging import ComponentLogger


@dataclass(frozen=True)
class DatasetSplit:
    timestamps: tuple[str, ...]
    features: tuple[tuple[float, ...], ...]
    labels: tuple[int, ...]


def load_refined_dataset(
    manifest_path: Path,
    refined_root: Path,
    logger: ComponentLogger,
    requested_splits: tuple[str, ...] | None = None,
) -> dict[str, DatasetSplit]:
    manifest = yaml.safe_load(manifest_path.read_text())
    schema = manifest["refined_schema"]
    checksums = manifest["checksums"]["refined"]
    dataset_dir = refined_root / manifest["id"]
    datasets = {}
    logger.info(f"Loading refined dataset: {manifest['id']}")

    # 4. resolve the splits required by the active protocol (documentation excerpt; keep synchronized)
    requested_splits = requested_splits or tuple(manifest["files"])
    missing_splits = set(requested_splits) - manifest["files"].keys()
    if missing_splits:
        message = f"Dataset manifest is missing splits: {sorted(missing_splits)}"
        logger.error(message)
        raise ValueError(message)
    # 5. verify and materialize each refined split (documentation excerpt; keep synchronized)
    for split in requested_splits:
        path = dataset_dir / f"{split}.csv"
        if sha256(path) != checksums[split]:
            message = f"Checksum mismatch for {path}"
            logger.error(message)
            raise ValueError(message)
        validate_csv(path, schema)
        datasets[split] = load_split(path, schema)
        logger.info(f"Loaded {len(datasets[split].features)} rows from {path}")

    return datasets


def load_split(path: Path, schema: dict[str, object]) -> DatasetSplit:
    timestamps = []
    features = []
    labels = []
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            timestamps.append(row[schema["timestamp"]])
            features.append(
                tuple(float(row[feature]) for feature in schema["features"])
            )
            labels.append(int(row[schema["label"]]))

    return DatasetSplit(tuple(timestamps), tuple(features), tuple(labels))
