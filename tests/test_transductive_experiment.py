import json
from pathlib import Path

import pytest
import yaml

from mark_ad_benchmark.datasets.ingest import sha256
from mark_ad_benchmark.experiment import Experiment
from mark_ad_benchmark.logging import Logger
from mark_ad_benchmark.models import RobustPCAModel


def test_transductive_experiment_uses_validation_and_test_without_train(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    refined_dir = tmp_path / "refined" / "example"
    refined_dir.mkdir(parents=True)
    write_split(refined_dir / "validation.csv", ((0, 0, 0), (0, 0, 0)))
    write_split(refined_dir / "test.csv", ((0, 0, 0), (0, 0, 1)))
    manifest_root = tmp_path / "manifests"
    manifest_root.mkdir()
    (manifest_root / "example.yaml").write_text(
        yaml.safe_dump(
            {
                "id": "example",
                "files": {
                    "train": "source/train.csv",
                    "validation": "source/validation.csv",
                    "test": "source/test.csv",
                },
                "refined_schema": {
                    "timestamp": "timestamp",
                    "features": ["feature_0", "feature_1"],
                    "label": "label",
                },
                "checksums": {
                    "refined": {
                        "validation": sha256(refined_dir / "validation.csv"),
                        "test": sha256(refined_dir / "test.csv"),
                    }
                },
            }
        )
    )
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(
        yaml.safe_dump(
            {
                "id": "example-robust-pca-transductive",
                "dataset": "example",
                "model": {
                    "name": "robust_pca",
                    "parameters": {
                        "tolerance": 1e-7,
                        "max_iterations": 10,
                    },
                    "candidates": {"lambda": [0.5, 0.25]},
                },
                "protocol": "transductive",
                "training": {"epochs": 1},
                "threshold": {"name": "validation_f1"},
                "selection": {"metric": "f1", "direction": "maximize"},
                "metrics": {
                    "vus": {"max_window": 2, "threshold_samples": 3},
                    "range": {"recall_existence_weight": 0.2},
                },
            }
        )
    )
    model_lambdas: list[float] = []
    original_fit_score = RobustPCAModel.fit_score

    def record_fit_score(self: RobustPCAModel, features: object) -> tuple[float, ...]:
        model_lambdas.append(self.lambda_value)
        return original_fit_score(self, features)

    monkeypatch.setattr(RobustPCAModel, "fit_score", record_fit_score)
    experiment = Experiment.from_path(
        experiment_path,
        Logger().for_component("mark_ad_benchmark.experiment"),
        manifest_root=manifest_root,
        refined_root=tmp_path / "refined",
        runs_root=tmp_path / "runs",
    )

    validated = experiment.train("validated-run")
    tested = experiment.test("validated-run", "test-run")

    validated_manifest = json.loads((validated.run_dir / "manifest.json").read_text())
    test_metrics = json.loads((tested.run_dir / "metrics.json").read_text())
    assert not (validated.run_dir / "model.pkl").exists()
    assert validated_manifest["protocol"]["execution_mode"] == "transductive"
    assert validated_manifest["model_diagnostics"]["iterations"] == 0
    assert test_metrics["model_diagnostics"]["iterations"] == 0
    assert validated_manifest["selected_model_parameters"]["lambda"] == 0.5
    assert model_lambdas == [0.5, 0.25, 0.5]


def write_split(path: Path, rows: tuple[tuple[int, int, int], ...]) -> None:
    lines = ["timestamp,feature_0,feature_1,label"]
    lines.extend(
        f"2025-01-{index:02d},{feature_0},{feature_1},{label}"
        for index, (feature_0, feature_1, label) in enumerate(rows, start=1)
    )
    path.write_text("\n".join(lines) + "\n")
