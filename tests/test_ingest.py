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
