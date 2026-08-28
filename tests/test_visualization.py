import csv
import json
from pathlib import Path

import pytest
import yaml

from mark_ad_benchmark.datasets.ingest import sha256
from mark_ad_benchmark.logging import ConsoleOutput, Logger
from mark_ad_benchmark.visualization import timeseries, visualize_runs


def test_visualization_exports_saved_test_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
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
        json.dumps(
            {
                "artifact_type": "validated",
                "experiment": {"dataset": "example"},
                "selected_threshold": 0.6,
                "threshold_state": {"threshold": 0.6},
            }
        )
    )
    test_dir = runs_root / "test-run"
    test_dir.mkdir()
    (test_dir / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_type": "test",
                "experiment": {
                    "id": "example-pca",
                    "model": {
                        "name": "pca",
                        "parameters": {"n_components": 1},
                        "candidates": {"n_components": [1, 2]},
                    },
                    "protocol": "inductive",
                    "threshold": {"name": "validation_f1"},
                    "selection": {"metric": "f1", "direction": "maximize"},
                },
                "validated_run": "validated-run",
                "selected_model_parameters": {"n_components": 1},
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
                "model_diagnostics": {"iterations": 1},
            }
        )
    )

    output_path = visualize_runs(
        Logger(ConsoleOutput()).for_component(
            "mark_ad_benchmark.visualization.timeseries"
        ),
        manifest_root,
        tmp_path / "refined",
        runs_root,
        tmp_path / "runs.json",
    )
    output = capsys.readouterr().out

    assert output_path == tmp_path / "runs.json"
    assert "[INFO] Exporting 1 saved test run" in output
    assert "[INFO] Exporting test run 1/1: test-run" in output
    assert "[INFO] Wrote documentation run data for 1 test run:" in output
    exported_run = json.loads(output_path.read_text())["test-run"]
    assert exported_run["display_name"] == "Test Run"
    assert exported_run["dataset_id"] == "example"
    assert exported_run["run_time"] is None
    assert exported_run["feature_labels"] == {"feature_0": "feature_0"}
    assert exported_run["configuration"] == {
        "model": "pca",
        "configured_parameters": {"n_components": 1},
        "candidates": {"n_components": [1, 2]},
        "selected_parameters": {"n_components": 1},
        "protocol": "inductive",
        "threshold_selector": "validation_f1",
        "validation_threshold": 0.6,
        "threshold_calibration": {"threshold": 0.6},
        "selection": {"metric": "f1", "direction": "maximize"},
        "model_diagnostics": {"iterations": 1},
    }


def test_visualization_sampling_retains_anomalies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(timeseries, "MAX_DISPLAY_POINTS", 3)

    indexes = timeseries.display_indexes((0, 0, 1, 0, 0, 0, 1, 0, 0, 0))

    assert len(indexes) <= 5
    assert 2 in indexes
    assert 6 in indexes
