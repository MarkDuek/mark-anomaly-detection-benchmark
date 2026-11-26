from pathlib import Path

import pytest

from mark_ad_benchmark.datasets.ingest import validate_csv


def test_validate_csv_accepts_declared_schema(tmp_path: Path) -> None:
    path = tmp_path / "series.csv"
    path.write_text("timestamp,in_count,out_count,is_anomaly\n2025-01-01,1,2,0\n")

    validate_csv(
        path,
        {
            "timestamp": "timestamp",
            "features": ["in_count", "out_count"],
            "label": "is_anomaly",
        },
    )


def test_validate_csv_rejects_invalid_labels(tmp_path: Path) -> None:
    path = tmp_path / "series.csv"
    path.write_text("timestamp,in_count,out_count,is_anomaly\n2025-01-01,1,2,2\n")

    with pytest.raises(ValueError, match="Invalid label"):
        validate_csv(
            path,
            {
                "timestamp": "timestamp",
                "features": ["in_count", "out_count"],
                "label": "is_anomaly",
            },
        )


def test_validate_csv_requires_explicit_normal_only_contract(tmp_path: Path) -> None:
    path = tmp_path / "train.csv"
    path.write_text("timestamp,value\n1,0.5\n")
    schema = {"timestamp": "timestamp", "features": ["value"], "label": "label"}

    validate_csv(path, schema, normal_only=True)
    with pytest.raises(ValueError, match="Unexpected columns"):
        validate_csv(path, schema)
    path.write_text("timestamp,value,label\n1,0.5,1\n")
    with pytest.raises(ValueError, match="Unexpected columns"):
        validate_csv(path, schema, normal_only=True)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_validate_csv_rejects_nonfinite_features(tmp_path: Path, value: str) -> None:
    path = tmp_path / "series.csv"
    path.write_text(f"timestamp,value,label\n1,{value},0\n")
    with pytest.raises(ValueError, match="Nonfinite feature"):
        validate_csv(
            path,
            {"timestamp": "timestamp", "features": ["value"], "label": "label"},
        )
