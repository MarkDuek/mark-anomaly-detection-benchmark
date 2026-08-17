from pathlib import Path

import pytest
import yaml

from mark_ad_benchmark.datasets.ingest import sha256
from mark_ad_benchmark.datasets.loader import load_refined_dataset
from mark_ad_benchmark.logging import Logger, LogOutput


class CapturingOutput(LogOutput):
    def __init__(self) -> None:
        self.messages: list[str] = []

    def log(self, message: str) -> None:
        self.messages.append(message)


def test_load_refined_dataset_returns_declared_splits(tmp_path: Path) -> None:
    refined_path = tmp_path / "refined" / "example" / "train.csv"
    refined_path.parent.mkdir(parents=True)
    refined_path.write_text("timestamp,feature_0,label\n2025-01-01,1.5,0\n")
    manifest_path = tmp_path / "dataset.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "id": "example",
                "files": {"train": "source/train.csv"},
                "refined_schema": {
                    "timestamp": "timestamp",
                    "features": ["feature_0"],
                    "label": "label",
                },
                "checksums": {"refined": {"train": sha256(refined_path)}},
            }
        )
    )

    output = CapturingOutput()
    datasets = load_refined_dataset(
        manifest_path,
        tmp_path / "refined",
        Logger(output).for_component("mark_ad_benchmark.datasets.loader"),
    )

    assert datasets["train"].timestamps == ("2025-01-01",)
    assert datasets["train"].features == ((1.5,),)
    assert datasets["train"].labels == (0,)
    assert output.messages == [
        "[INFO] Loading refined dataset: example",
        f"[INFO] Loaded 1 rows from {refined_path}",
    ]


def test_load_refined_dataset_rejects_changed_file(tmp_path: Path) -> None:
    refined_path = tmp_path / "refined" / "example" / "train.csv"
    refined_path.parent.mkdir(parents=True)
    refined_path.write_text("timestamp,feature_0,label\n2025-01-01,1.5,0\n")
    manifest_path = tmp_path / "dataset.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "id": "example",
                "files": {"train": "source/train.csv"},
                "refined_schema": {
                    "timestamp": "timestamp",
                    "features": ["feature_0"],
                    "label": "label",
                },
                "checksums": {"refined": {"train": "wrong"}},
            }
        )
    )

    output = CapturingOutput()
    with pytest.raises(ValueError, match="Checksum mismatch"):
        load_refined_dataset(
            manifest_path,
            tmp_path / "refined",
            Logger(output).for_component("mark_ad_benchmark.datasets.loader"),
        )

    assert output.messages == [
        "[INFO] Loading refined dataset: example",
        f"[ERROR] Checksum mismatch for {refined_path}",
    ]
