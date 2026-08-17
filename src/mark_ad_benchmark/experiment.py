import csv
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path

import yaml

from mark_ad_benchmark.datasets.loader import DatasetSplit, load_refined_dataset
from mark_ad_benchmark.logging import ComponentLogger
from mark_ad_benchmark.metrics import PointMetrics, point_metrics, select_f1_threshold


@dataclass(frozen=True)
class TrainingResult:
    run_dir: Path
    threshold: float
    validation_metrics: PointMetrics


@dataclass(frozen=True)
class TestResult:
    run_dir: Path
    threshold: float
    test_metrics: PointMetrics


@dataclass(frozen=True)
class ExperimentResult:
    training: TrainingResult
    test: TestResult


def train_experiment(
    experiment_path: Path,
    logger: ComponentLogger,
    manifest_root: Path = Path("data/manifests"),
    refined_root: Path = Path("data/refined"),
    runs_root: Path = Path("runs"),
    run_id: str | None = None,
    config_root: Path = Path("config"),
) -> TrainingResult:
    """Fit and calibrate an experiment without loading its test split."""
    experiment = load_experiment(experiment_path)
    protocol_type, protocol_module = resolve_available_component(
        config_root / "protocols.yaml", experiment["protocol"], "protocol", logger
    )
    model_type, model_module = resolve_available_component(
        config_root / "models.yaml", experiment["model"]["name"], "model", logger
    )
    model = model_type(
        experiment["model"]["n_components"], logger.for_component(model_module)
    )
    manifest_path, dataset_manifest = load_dataset_manifest(
        experiment, manifest_root, logger
    )
    splits = load_refined_dataset(
        manifest_path,
        refined_root,
        logger.for_component("mark_ad_benchmark.datasets.loader"),
        protocol_type.training_required_splits,
    )
    protocol = protocol_type.from_splits(splits, logger.for_component(protocol_module))
    validation_scores = protocol.fit_and_score_validation(model)
    metrics_logger = logger.for_component("mark_ad_benchmark.metrics.point")
    threshold = select_f1_threshold(
        protocol.validation.labels, validation_scores, metrics_logger
    )
    validation_metrics = point_metrics(
        protocol.validation.labels, validation_scores, threshold, metrics_logger
    )

    training_run_id = run_id or create_run_id(f"{experiment['id']}-train")
    run_dir = runs_root / training_run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    model.save(run_dir / "model.pkl")
    write_scores(
        run_dir / "validation_scores.csv",
        "validation",
        protocol.validation,
        validation_scores,
        threshold,
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {"threshold": threshold, "validation": asdict(validation_metrics)},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    write_manifest(
        run_dir / "manifest.json",
        {
            "artifact_type": "training",
            "dataset_manifest": dataset_manifest,
            "experiment": experiment,
            "model": {"module": model_module, "class": model_type.__name__},
            "protocol": {"module": protocol_module, "class": protocol_type.__name__},
            "threshold": threshold,
        },
    )
    logger.info(
        f"Completed training: validation_f1={validation_metrics.f1}, run_dir={run_dir}"
    )
    return TrainingResult(run_dir, threshold, validation_metrics)


def test_experiment(
    experiment_path: Path,
    training_run_id: str,
    logger: ComponentLogger,
    manifest_root: Path = Path("data/manifests"),
    refined_root: Path = Path("data/refined"),
    runs_root: Path = Path("runs"),
    run_id: str | None = None,
    config_root: Path = Path("config"),
) -> TestResult:
    """Score test data using a saved training artifact without fitting."""
    experiment = load_experiment(experiment_path)
    training_dir = runs_root / training_run_id
    training_manifest_path = training_dir / "manifest.json"
    if not training_manifest_path.is_file():
        message = f"Training artifact not found: {training_dir}"
        logger.error(message)
        raise FileNotFoundError(message)
    training_manifest = json.loads(training_manifest_path.read_text())
    if training_manifest["artifact_type"] != "training":
        message = f"Artifact is not a training run: {training_dir}"
        logger.error(message)
        raise ValueError(message)
    if training_manifest["experiment"] != experiment:
        message = "Training artifact does not match the experiment definition"
        logger.error(message)
        raise ValueError(message)

    protocol_type, protocol_module = resolve_available_component(
        config_root / "protocols.yaml", experiment["protocol"], "protocol", logger
    )
    model_type, model_module = resolve_available_component(
        config_root / "models.yaml", experiment["model"]["name"], "model", logger
    )
    if training_manifest["model"] != {
        "module": model_module,
        "class": model_type.__name__,
    } or training_manifest["protocol"] != {
        "module": protocol_module,
        "class": protocol_type.__name__,
    }:
        message = "Training artifact components do not match the available catalogs"
        logger.error(message)
        raise ValueError(message)

    manifest_path, dataset_manifest = load_dataset_manifest(
        experiment, manifest_root, logger
    )
    if training_manifest["dataset_manifest"] != dataset_manifest:
        message = "Training artifact does not match the dataset manifest"
        logger.error(message)
        raise ValueError(message)
    model = model_type.load(
        training_dir / "model.pkl", logger.for_component(model_module)
    )
    splits = load_refined_dataset(
        manifest_path,
        refined_root,
        logger.for_component("mark_ad_benchmark.datasets.loader"),
        protocol_type.test_required_splits,
    )
    test_split = splits["test"]
    test_scores = protocol_type.score_test(
        model, test_split, logger.for_component(protocol_module)
    )
    threshold = training_manifest["threshold"]
    metrics_logger = logger.for_component("mark_ad_benchmark.metrics.point")
    test_metrics = point_metrics(
        test_split.labels, test_scores, threshold, metrics_logger
    )

    test_run_id = run_id or create_run_id(f"{experiment['id']}-test")
    run_dir = runs_root / test_run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    write_scores(
        run_dir / "test_scores.csv", "test", test_split, test_scores, threshold
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {"threshold": threshold, "test": asdict(test_metrics)},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    write_manifest(
        run_dir / "manifest.json",
        {
            "artifact_type": "test",
            "experiment": experiment,
            "training_run": training_run_id,
            "threshold": threshold,
        },
    )
    logger.info(f"Completed testing: test_f1={test_metrics.f1}, run_dir={run_dir}")
    return TestResult(run_dir, threshold, test_metrics)


def run_experiment(
    experiment_path: Path,
    logger: ComponentLogger,
    manifest_root: Path = Path("data/manifests"),
    refined_root: Path = Path("data/refined"),
    runs_root: Path = Path("runs"),
    run_id: str | None = None,
    config_root: Path = Path("config"),
) -> ExperimentResult:
    """Train then test a manifest-defined experiment with separate artifacts."""
    training = train_experiment(
        experiment_path,
        logger,
        manifest_root,
        refined_root,
        runs_root,
        f"{run_id}-train" if run_id else None,
        config_root,
    )
    test = test_experiment(
        experiment_path,
        training.run_dir.name,
        logger,
        manifest_root,
        refined_root,
        runs_root,
        f"{run_id}-test" if run_id else None,
        config_root,
    )
    return ExperimentResult(training, test)


def load_experiment(experiment_path: Path) -> dict[str, object]:
    return yaml.safe_load(experiment_path.read_text())


def load_dataset_manifest(
    experiment: dict[str, object], manifest_root: Path, logger: ComponentLogger
) -> tuple[Path, dict[str, object]]:
    manifest_path = manifest_root / f"{experiment['dataset']}.yaml"
    if not manifest_path.is_file():
        message = f"Dataset manifest not found: {manifest_path}"
        logger.error(message)
        raise FileNotFoundError(message)
    return manifest_path, yaml.safe_load(manifest_path.read_text())


def resolve_available_component(
    catalog_path: Path,
    name: str,
    component_type: str,
    logger: ComponentLogger,
) -> tuple[object, str]:
    catalog = yaml.safe_load(catalog_path.read_text())
    component = catalog.get(name)
    if component is None:
        message = f"Unsupported {component_type}: {name}"
        logger.error(message)
        raise ValueError(message)
    module_name = component["module"]
    return getattr(import_module(module_name), component["class"]), module_name


def create_run_id(experiment_id: str) -> str:
    return f"{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}-{experiment_id}"


def write_manifest(path: Path, content: dict[str, object]) -> None:
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n")


def write_scores(
    path: Path,
    split_name: str,
    split: DatasetSplit,
    scores: tuple[float, ...],
    threshold: float,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["split", "timestamp", "label", "score", "prediction"],
            lineterminator="\n",
        )
        writer.writeheader()
        for timestamp, label, score in zip(
            split.timestamps, split.labels, scores, strict=True
        ):
            writer.writerow(
                {
                    "split": split_name,
                    "timestamp": timestamp,
                    "label": label,
                    "score": score,
                    "prediction": int(score >= threshold),
                }
            )
