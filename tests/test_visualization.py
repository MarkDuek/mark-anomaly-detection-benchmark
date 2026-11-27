import csv
import json
from pathlib import Path

import pytest
import yaml

from mark_ad_benchmark.datasets.ingest import sha256
from mark_ad_benchmark.logging import ConsoleOutput, Logger
from mark_ad_benchmark.metrics.ranking import ranking_curves
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
                "test": {
                    "point": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
                    "ranking": {"auc_roc": 1.0, "auc_pr": 1.0},
                },
                "model_diagnostics": {"iterations": 1},
            }
        )
    )
    (validated_dir / "metrics.json").write_text(
        json.dumps(
            {
                "threshold": 0.6,
                "validation": {"ranking": {"auc_roc": 1.0, "auc_pr": 1.0}},
            }
        )
    )
    for directory, threshold in ((validated_dir, 0.6), (test_dir, 0.5)):
        (directory / "ranking_curves.json").write_text(
            json.dumps(ranking_curves((0, 1), (0.2, 0.8), threshold))
        )

    assert (
        timeseries.visualize_training(
            validated_dir,
            Logger().for_component("mark_ad_benchmark.visualization.training"),
        )
        is None
    )
    (validated_dir / "training_history.json").write_text(
        json.dumps(
            [
                {
                    "iteration": iteration,
                    "estimated_entropy": 2.0 + iteration,
                    "projected_gradient_norm": 0.1,
                    "subspace_change": 0.01,
                    "bandwidth_squared": 1.0,
                }
                for iteration in (0, 10)
            ]
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
    assert exported_run["curves_url"] == "run-curves/test-run.json"
    assert exported_run["training_url"] == "run-training/validated-run.html"
    training_html = (validated_dir / "training.html").read_text()
    assert "Estimated entropy" in training_html
    assert "Monitoring tangent-gradient norm" in training_html
    assert "Adaptive bandwidth" in training_html
    assert (tmp_path / exported_run["training_url"]).read_text() == training_html
    assert "curves" not in exported_run
    curves = json.loads((tmp_path / exported_run["curves_url"]).read_text())
    assert curves["validation"]["threshold"] == 0.6
    assert curves["test"]["threshold"] == 0.5
    assert curves["validation"]["curves"]["roc"]["tpr"][-1] == 1.0
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

    (validated_dir / "ranking_curves.json").unlink()
    with pytest.raises(
        FileNotFoundError, match="Ranking curves not found: .*regenerate"
    ):
        visualize_runs(
            Logger().for_component("mark_ad_benchmark.visualization.timeseries"),
            manifest_root,
            tmp_path / "refined",
            runs_root,
            tmp_path / "missing" / "runs.json",
        )
    assert not (tmp_path / "missing" / "runs.json").exists()


def test_publish_packages_selected_generated_runs(tmp_path: Path) -> None:
    source_root = tmp_path / "local"
    curves_dir = source_root / "run-curves"
    training_dir = source_root / "run-training"
    curves_dir.mkdir(parents=True)
    training_dir.mkdir()
    runs = {
        "published": {
            "curves_url": "run-curves/published.json",
            "training_url": "run-training/validated.html",
        },
        "private": {"curves_url": "run-curves/private.json"},
    }
    (source_root / "runs.json").write_text(json.dumps(runs))
    (curves_dir / "published.json").write_text("published curves")
    (curves_dir / "private.json").write_text("private curves")
    (training_dir / "validated.html").write_text("training")
    published_runs_path = tmp_path / "published-runs.txt"
    published_runs_path.write_text("# Approved runs\npublished\n")
    output_path = tmp_path / "published-runs"
    output_path.mkdir()
    (output_path / "stale.txt").write_text("stale")

    result = timeseries.publish_runs(
        Logger().for_component("mark_ad_benchmark.visualization.timeseries"),
        published_runs_path,
        source_root / "runs.json",
        output_path,
    )

    assert result == output_path
    assert json.loads((output_path / "runs.json").read_text()) == {
        "published": runs["published"]
    }
    assert (output_path / "run-curves/published.json").read_text() == "published curves"
    assert (output_path / "run-training/validated.html").read_text() == "training"
    assert not (output_path / "run-curves/private.json").exists()
    assert not (output_path / "stale.txt").exists()

    published_runs_path.write_text("# No published runs\n")
    empty_output_path = tmp_path / "empty-published-runs"
    timeseries.publish_runs(
        Logger().for_component("mark_ad_benchmark.visualization.timeseries"),
        published_runs_path,
        tmp_path / "missing.json",
        empty_output_path,
    )

    assert json.loads((empty_output_path / "runs.json").read_text()) == {}


def test_visualization_sampling_retains_anomalies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(timeseries, "MAX_DISPLAY_POINTS", 3)

    indexes = timeseries.display_indexes((0, 0, 1, 0, 0, 0, 1, 0, 0, 0))

    assert len(indexes) <= 5
    assert 2 in indexes
    assert 6 in indexes


def test_curve_export_keeps_corners_and_saved_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    curves = {
        "roc": {
            "fpr": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1],
            "tpr": [0, 0.5, 0.5, 0.5, 0.5, 1, 1],
        },
        "pr": {
            "recall": [1, 1, 0.5, 0.5, 0.5, 0, 0],
            "precision": [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        },
        "operating_point": {"fpr": 0.2, "tpr": 0.5, "recall": 0.5, "precision": 0.8},
    }
    (tmp_path / "ranking_curves.json").write_text(json.dumps(curves))
    (tmp_path / "metrics.json").write_text(
        json.dumps({"threshold": 10, "test": {"ranking": {"auc_roc": 0.8}}})
    )
    logger = Logger().for_component("mark_ad_benchmark.visualization.timeseries")

    exported = timeseries.load_run_curves(tmp_path, "test", logger)
    assert exported["curves"]["roc"] == {
        key: [points[index] for index in (0, 1, 4, 5, 6)]
        for key, points in curves["roc"].items()
    }
    assert exported["curves"]["pr"] == {
        key: [points[index] for index in (0, 1, 2, 4, 5, 6)]
        for key, points in curves["pr"].items()
    }
    assert exported["curves"]["operating_point"] == curves["operating_point"]
    assert exported["ranking"] == {"auc_roc": 0.8}
    assert json.loads((tmp_path / "ranking_curves.json").read_text()) == curves

    monkeypatch.setattr(timeseries, "MAX_DISPLAY_POINTS", 3)
    capped = timeseries.load_run_curves(tmp_path, "test", logger)["curves"]
    assert len(capped["roc"]["fpr"]) == len(capped["pr"]["recall"]) == 3
    assert capped["roc"]["fpr"][0] == 0
    assert capped["roc"]["fpr"][-1] == 1
