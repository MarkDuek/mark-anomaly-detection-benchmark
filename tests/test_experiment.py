import json
from pathlib import Path

import pytest
import yaml

from mark_ad_benchmark.datasets.ingest import sha256
from mark_ad_benchmark.experiment import Experiment
from mark_ad_benchmark.logging import ConsoleOutput, Logger


@pytest.mark.parametrize("model_name", ["pca", "kernel_pca", "convolutional_pca"])
def test_inductive_training_and_testing_use_separate_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], model_name: str
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
                    "name": model_name,
                    "parameters": {
                        "n_components": 1,
                        "standardize_features": True,
                        **{
                            "pca": {},
                            "kernel_pca": {"kernel": "linear"},
                            "convolutional_pca": {"filter_order": 0},
                        }[model_name],
                    },
                },
                "protocol": "inductive",
                "training": {"epochs": 1},
                "threshold": {"name": "validation_f1"},
                "selection": {"metric": "f1", "direction": "maximize"},
                "metrics": metric_definition(),
            }
        )
    )

    experiment = Experiment.from_path(
        experiment_path,
        Logger(ConsoleOutput()).for_component("mark_ad_benchmark.experiment"),
        manifest_root=manifest_root,
        refined_root=tmp_path / "refined",
        runs_root=tmp_path / "runs",
    )
    validated = experiment.train("validated-run")
    output = capsys.readouterr().out

    assert validated.validation.epoch == 1
    assert (
        "[INFO] ==================== TRAINING AND VALIDATION ===================="
        in output
    )
    assert "[INFO] Loading training and validation data" in output
    assert "[INFO] Training candidate 1/1, epoch 1/1:" in output
    assert "[INFO] Validating 1 parameter candidate across 1 epoch" in output
    assert "[INFO] Validating candidate 1/1, epoch 1/1:" in output
    assert "[INFO] Completed validation phase:" in output
    assert validated.validation.threshold == pytest.approx(3**0.5)
    assert (
        validated.validation.metrics.point.precision,
        validated.validation.metrics.point.recall,
        validated.validation.metrics.point.f1,
    ) == pytest.approx((1.0, 1.0, 1.0))
    assert (validated.run_dir / "model.pkl").is_file()
    assert (validated.run_dir / "validation_scores.csv").is_file()
    validation_curves = json.loads(
        (validated.run_dir / "ranking_curves.json").read_text()
    )
    assert validation_curves["operating_point"]["precision"] == pytest.approx(1.0)
    validation_metrics = json.loads((validated.run_dir / "metrics.json").read_text())[
        "validation"
    ]
    assert set(validation_metrics) == {
        "point",
        "event",
        "range",
        "affiliation",
        "ranking",
        "vus_ranking",
        "vus_prediction",
    }
    assert validation_metrics["ranking"]["auc_roc"] == pytest.approx(1.0)
    assert (
        json.loads((validated.run_dir / "validation_history.json").read_text())[0][
            "epoch"
        ]
        == 1
    )

    testing = experiment.test("validated-run", "test-run")
    output = capsys.readouterr().out

    assert testing.threshold == validated.validation.threshold
    assert "[INFO] ========================= TEST =========================" in output
    assert "[INFO] Loading test data" in output
    assert "[INFO] Scoring test data with selected validation configuration:" in output
    assert "[INFO] Completed test phase:" in output
    assert (
        testing.test_metrics.point.precision,
        testing.test_metrics.point.recall,
        testing.test_metrics.point.f1,
    ) == pytest.approx((1.0, 1.0, 1.0))
    assert (testing.run_dir / "test_scores.csv").is_file()
    test_curves = json.loads((testing.run_dir / "ranking_curves.json").read_text())
    assert test_curves["operating_point"]["recall"] == pytest.approx(1.0)
    assert (testing.run_dir / "metrics.json").is_file()
    assert (testing.run_dir / "manifest.json").is_file()
    test_metrics = json.loads((testing.run_dir / "metrics.json").read_text())["test"]
    assert set(test_metrics) == set(validation_metrics)
    assert test_metrics["event"] == {
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }
    validation_diagnostics = json.loads(
        (validated.run_dir / "metrics.json").read_text()
    )["model_diagnostics"]
    test_diagnostics = json.loads((testing.run_dir / "metrics.json").read_text())[
        "model_diagnostics"
    ]
    assert test_diagnostics == validation_diagnostics
    if model_name == "convolutional_pca":
        assert test_diagnostics["converged"] is True
    else:
        assert test_diagnostics == {}


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
                "metrics": metric_definition(),
            }
        )
    )

    with pytest.raises(ValueError, match=message):
        Experiment.from_path(
            experiment_path,
            Logger().for_component("mark_ad_benchmark.experiment"),
            manifest_root=tmp_path / "missing",
        )


def test_explicit_parameter_candidates_override_base_parameters() -> None:
    experiment = Experiment.from_path(
        Path("experiments/calit2-kernel-pca-inductive.yaml"),
        Logger().for_component("mark_ad_benchmark.experiment"),
    )
    experiment.definition["model"]["candidates"] = [
        {"kernel": "linear", "n_components": 2},
        {"kernel": "rbf", "kernel_parameters": {"gamma": 0.5}},
    ]
    experiment.validate_definition()

    assert experiment.model_parameter_candidates() == (
        {
            "n_components": 2,
            "standardize_features": True,
            "kernel_parameters": {},
            "kernel": "linear",
        },
        {
            "n_components": 1,
            "standardize_features": True,
            "kernel": "rbf",
            "kernel_parameters": {"gamma": 0.5},
        },
    )


@pytest.mark.parametrize(
    "candidates", [[], ["linear"], [{1: "linear"}], None, {"kernel": []}]
)
def test_rejects_invalid_parameter_candidates(candidates: object) -> None:
    experiment = Experiment.from_path(
        Path("experiments/calit2-kernel-pca-inductive.yaml"),
        Logger().for_component("mark_ad_benchmark.experiment"),
    )
    experiment.definition["model"]["candidates"] = candidates

    with pytest.raises(TypeError, match="model.candidates"):
        experiment.validate_definition()


def test_creditcard_tuning_candidates() -> None:
    logger = Logger().for_component("mark_ad_benchmark.experiment")
    fixed = Experiment.from_path(
        Path("experiments/creditcard-pca-inductive-validation-f1.yaml"), logger
    )
    quantile = Experiment.from_path(
        Path("experiments/creditcard-pca-inductive-validation-f1-quantile.yaml"),
        logger,
    )
    assert {
        **fixed.definition,
        "id": quantile.definition["id"],
        "threshold": quantile.definition["threshold"],
    } == quantile.definition
    assert [
        candidate["n_components"] for candidate in fixed.model_parameter_candidates()
    ] == [
        13,
        20,
        26,
    ]

    nystrom = Experiment.from_path(
        Path(
            "experiments/creditcard-nystrom-kpca-inductive-kernels-validation-f1-quantile.yaml"
        ),
        logger,
    )
    candidates = nystrom.model_parameter_candidates()
    assert {
        kernel: [
            candidate["n_components"]
            for candidate in candidates
            if candidate["kernel"] == kernel
        ]
        for kernel in ("linear", "poly", "rbf")
    } == {
        "linear": [5, 13, 20, 26],
        "poly": [5, 13, 20, 26],
        "rbf": [5, 13, 20, 26, 32, 48],
    }
    for candidate in candidates:
        nystrom.create_model(candidate)


def write_split(path: Path, rows: tuple[tuple[int, int, int], ...]) -> None:
    lines = ["timestamp,feature_0,feature_1,label"]
    lines.extend(
        f"2025-01-{index:02d},{feature_0},{feature_1},{label}"
        for index, (feature_0, feature_1, label) in enumerate(rows, start=1)
    )
    path.write_text("\n".join(lines) + "\n")


def metric_definition() -> dict[str, object]:
    return {
        "vus": {"max_window": 2, "threshold_samples": 3},
        "range": {"recall_existence_weight": 0.2},
    }
