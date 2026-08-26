import csv
import json
from pathlib import Path

import pytest
import yaml

from mark_ad_benchmark.datasets.ingest import sha256
from mark_ad_benchmark.logging import Logger
from mark_ad_benchmark.visualization import timeseries, visualize_runs


def test_visualization_dashboard_embeds_saved_test_run(tmp_path: Path) -> None:
    refined_dir = tmp_path / "refined" / "example"
    refined_dir.mkdir(parents=True)
    split_path = refined_dir / "test.csv"
    split_path.write_text("timestamp,feature_0,label\n2025-01-01,1,0\n2025-01-02,3,1\n")
    manifest_root = tmp_path / "manifests"
    manifest_root.mkdir()
    (manifest_root / "example.yaml").write_text(
        yaml.safe_dump(
            {
                "id": "example",
                "files": {"test": "source/test.csv"},
                "raw_schema": {
                    "timestamp": "timestamp",
                    "features": ["feature_0"],
                    "label": "label",
                },
                "refined_schema": {
                    "timestamp": "timestamp",
                    "features": ["feature_0"],
                    "label": "label",
                },
                "checksums": {"refined": {"test": sha256(split_path)}},
            }
        )
    )
    runs_root = tmp_path / "runs"
    validated_dir = runs_root / "validated-run"
    validated_dir.mkdir(parents=True)
    (validated_dir / "manifest.json").write_text(
        json.dumps({"artifact_type": "validated", "experiment": {"dataset": "example"}})
    )
    test_dir = runs_root / "test-run"
    test_dir.mkdir()
    (test_dir / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_type": "test",
                "experiment": {"id": "example-pca"},
                "validated_run": "validated-run",
            }
        )
    )
    with (test_dir / "test_scores.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["timestamp", "label", "score"])
        writer.writeheader()
        writer.writerows(
            [
                {"timestamp": "2025-01-01", "label": 0, "score": 0.2},
                {"timestamp": "2025-01-02", "label": 1, "score": 0.8},
            ]
        )
    (test_dir / "metrics.json").write_text(
        json.dumps(
            {
                "threshold": 0.5,
                "test": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
            }
        )
    )

    output_path = visualize_runs(
        Logger().for_component("mark_ad_benchmark.visualization.timeseries"),
        manifest_root,
        tmp_path / "refined",
        runs_root,
        tmp_path / "runs.json",
    )

    content = output_path.read_text()
    assert output_path == runs_root / "visualization.html"
    assert "test-run" in content
    assert "Anomaly score" in content
    assert "Prediction" in content
    assert 'id="metric-cards"' in content
    assert 'id="comparison-runs"' in content
    assert 'id="comparison-table"' in content
    assert (tmp_path / "runs.json").is_file()
    exported_run = json.loads((tmp_path / "runs.json").read_text())["test-run"]
    assert exported_run["display_name"] == "Test Run"
    assert exported_run["dataset_id"] == "example"
    assert exported_run["run_time"] is None
    assert exported_run["feature_labels"] == {"feature_0": "feature_0"}


def test_visualization_sampling_retains_anomalies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(timeseries, "MAX_DISPLAY_POINTS", 3)

    indexes = timeseries.display_indexes((0, 0, 1, 0, 0, 0, 1, 0, 0, 0))

    assert len(indexes) <= 5
    assert 2 in indexes
    assert 6 in indexes
