from pathlib import Path

import pytest
import yaml

from mark_ad_benchmark.datasets.ingest import sha256
from mark_ad_benchmark.datasets.refine import refine_dataset


def test_refine_dataset_writes_canonical_columns(tmp_path: Path) -> None:
    manifest_path = tmp_path / "dataset.yaml"
    raw_path = tmp_path / "raw" / "example" / "train.csv"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text("time,first,second,Label\n2025-01-01,1,2,0\n")
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "id": "example",
                "files": {"train": "source/train.csv"},
                "raw_schema": {
                    "timestamp": "time",
                    "features": ["first", "second"],
                    "label": "Label",
                },
                "refined_schema": {
                    "timestamp": "timestamp",
                    "features": ["feature_0", "feature_1"],
                    "label": "label",
                },
                "checksums": {"raw": {"train": sha256(raw_path)}},
            }
        )
    )

    refine_dataset(manifest_path, tmp_path / "raw", tmp_path / "refined")

    refined_path = tmp_path / "refined" / "example" / "train.csv"
    assert (
        refined_path.read_text()
        == "timestamp,feature_0,feature_1,label\n2025-01-01,1,2,0\n"
    )
    assert yaml.safe_load(manifest_path.read_text())["checksums"]["refined"][
        "train"
    ] == sha256(refined_path)


def test_refine_dataset_rejects_changed_raw_file(tmp_path: Path) -> None:
    manifest_path = tmp_path / "dataset.yaml"
    raw_path = tmp_path / "raw" / "example" / "train.csv"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text("time,first,Label\n2025-01-01,1,0\n")
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "id": "example",
                "files": {"train": "source/train.csv"},
                "raw_schema": {
                    "timestamp": "time",
                    "features": ["first"],
                    "label": "Label",
                },
                "refined_schema": {
                    "timestamp": "timestamp",
                    "features": ["feature_0"],
                    "label": "label",
                },
                "checksums": {"raw": {"train": "wrong"}},
            }
        )
    )

    with pytest.raises(ValueError, match="Checksum mismatch"):
        refine_dataset(manifest_path, tmp_path / "raw", tmp_path / "refined")
