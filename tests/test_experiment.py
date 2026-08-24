import json
from pathlib import Path

import pytest
import yaml

from mark_ad_benchmark.datasets.ingest import sha256
from mark_ad_benchmark.experiment import Experiment
from mark_ad_benchmark.logging import Logger


def test_pca_training_and_testing_use_separate_artifacts(
    tmp_path: Path,
) -> None:
    refined_dir = tmp_path / "refined" / "example"
    refined_dir.mkdir(parents=True)
    write_split(refined_dir / "train.csv", ((0, 0, 0), (1, 1, 0), (2, 2, 0)))
    write_split(refined_dir / "validation.csv", ((3, 3, 0), (1, 3, 1)))
    write_split(refined_dir / "test.csv", ((2, 2, 0), (0, 2, 1)))
    manifest_root = tmp_path / "manifests"
    manifest_root.mkdir()
    manifest_path = manifest_root / "example.yaml"
    manifest_path.write_text(
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
                        split: sha256(refined_dir / f"{split}.csv")
                        for split in ("train", "validation", "test")
                    }
                },
            }
        )
    )
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(
        yaml.safe_dump(
            {
                "id": "example-pca-inductive",
                "dataset": "example",
                "model": {
                    "name": "pca",
                    "parameters": {"n_components": 1, "feature_scaling": "standard"},
                },
                "protocol": "inductive",
                "training": {"epochs": 1},
                "threshold": {"name": "validation_f1"},
                "selection": {"metric": "f1", "direction": "maximize"},
            }
        )
    )

    experiment = Experiment.from_path(
        experiment_path,
        Logger().for_component("mark_ad_benchmark.experiment"),
        manifest_root=manifest_root,
        refined_root=tmp_path / "refined",
        runs_root=tmp_path / "runs",
    )
    validated = experiment.train("validated-run")

    assert validated.validation.epoch == 1
    assert validated.validation.threshold == pytest.approx(3**0.5)
    assert (
        validated.validation.metrics.precision,
        validated.validation.metrics.recall,
        validated.validation.metrics.f1,
    ) == pytest.approx((1.0, 1.0, 1.0))
    assert (validated.run_dir / "model.pkl").is_file()
    assert (validated.run_dir / "validation_scores.csv").is_file()
    assert (
        json.loads((validated.run_dir / "validation_history.json").read_text())[0][
            "epoch"
        ]
        == 1
    )

    testing = experiment.test("validated-run", "test-run")

    assert testing.threshold == validated.validation.threshold
    assert (
        testing.test_metrics.precision,
        testing.test_metrics.recall,
        testing.test_metrics.f1,
    ) == pytest.approx((1.0, 1.0, 1.0))
    assert (testing.run_dir / "test_scores.csv").is_file()
    assert (testing.run_dir / "metrics.json").is_file()
    assert (testing.run_dir / "manifest.json").is_file()


@pytest.mark.parametrize(
    ("model_name", "protocol_name", "message"),
    [
        ("missing", "inductive", "Unsupported model: missing"),
        ("pca", "missing", "Unsupported protocol: missing"),
    ],
)
def test_experiment_rejects_unknown_model_or_protocol_before_loading_data(
    tmp_path: Path,
    model_name: str,
    protocol_name: str,
    message: str,
) -> None:
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(
        yaml.safe_dump(
            {
                "id": "invalid",
                "dataset": "missing",
                "model": {
                    "name": model_name,
                    "parameters": {"n_components": 1},
                },
                "protocol": protocol_name,
                "training": {"epochs": 1},
                "threshold": {"name": "validation_f1"},
                "selection": {"metric": "f1", "direction": "maximize"},
            }
        )
    )

    with pytest.raises(ValueError, match=message):
        Experiment.from_path(
            experiment_path,
            Logger().for_component("mark_ad_benchmark.experiment"),
            manifest_root=tmp_path / "missing",
        )


def write_split(path: Path, rows: tuple[tuple[int, int, int], ...]) -> None:
    lines = ["timestamp,feature_0,feature_1,label"]
    lines.extend(
        f"2025-01-{index:02d},{feature_0},{feature_1},{label}"
        for index, (feature_0, feature_1, label) in enumerate(rows, start=1)
    )
    path.write_text("\n".join(lines) + "\n")
