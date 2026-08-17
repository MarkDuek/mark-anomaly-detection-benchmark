import csv
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path

import yaml

from mark_ad_benchmark.datasets.loader import DatasetSplit, load_refined_dataset
from mark_ad_benchmark.logging import ComponentLogger
from mark_ad_benchmark.metrics import PointMetrics, point_metrics


@dataclass(frozen=True)
class ValidationResult:
    epoch: int
    threshold: float
    metrics: PointMetrics
    selection_value: float
    scores: tuple[float, ...]


@dataclass(frozen=True)
class ValidatedRun:
    run_dir: Path
    validation: ValidationResult


@dataclass(frozen=True)
class TestResult:
    run_dir: Path
    threshold: float
    test_metrics: PointMetrics


@dataclass(frozen=True)
class ExperimentResult:
    validated: ValidatedRun
    test: TestResult


class Experiment:
    """Run one versioned experiment definition through training and evaluation."""

    def __init__(
        self,
        definition: dict[str, object],
        logger: ComponentLogger,
        manifest_root: Path = Path("data/manifests"),
        refined_root: Path = Path("data/refined"),
        runs_root: Path = Path("runs"),
        config_root: Path = Path("config"),
    ) -> None:
        self.definition = definition
        self.logger = logger
        self.manifest_root = manifest_root
        self.refined_root = refined_root
        self.runs_root = runs_root
        self.config_root = config_root
        self.validate_definition()
        self.protocol_type, self.protocol_module = resolve_available_component(
            config_root / "protocols.yaml", definition["protocol"], "protocol", logger
        )
        self.model_type, self.model_module = resolve_available_component(
            config_root / "models.yaml", definition["model"]["name"], "model", logger
        )
        self.threshold_type, self.threshold_module = resolve_available_component(
            config_root / "thresholds.yaml",
            definition["threshold"]["name"],
            "threshold selector",
            logger,
        )

    @classmethod
    def from_path(
        cls, experiment_path: Path, logger: ComponentLogger, **kwargs: Path
    ) -> "Experiment":
        """Create an experiment from a checked-in YAML definition."""
        return cls(yaml.safe_load(experiment_path.read_text()), logger, **kwargs)

    def train(self, run_id: str | None = None) -> ValidatedRun:
        """Train and validate every epoch, then persist the selected model."""
        manifest_path, dataset_manifest = self.load_dataset_manifest()

        # 1. load refined dataset splits
        splits = load_refined_dataset(
            manifest_path,
            self.refined_root,
            self.logger.for_component("mark_ad_benchmark.datasets.loader"),
            self.protocol_type.training_required_splits,
        )

        # 2. instantiate protocol from data splits
        protocol = self.protocol_type.from_splits(
            splits, self.logger.for_component(self.protocol_module)
        )

        # 3. instantiate model
        model = self.model_type(
            self.definition["model"]["parameters"],
            self.logger.for_component(self.model_module),
        )

        # 4. run set up
        validated_run_id = run_id or create_run_id(f"{self.definition['id']}-validated")
        self.runs_root.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.runs_root / f".{validated_run_id}.checkpoint"
        best_validation: ValidationResult | None = None
        history: list[dict[str, object]] = []

        # 5. training loop
        for epoch in range(1, self.definition["training"]["epochs"] + 1):
            self.logger.info(f"Training epoch {epoch}")

            # 5.1 fit model on training data
            protocol.fit(model)
            validation = self.validate(model, protocol, epoch)
            history.append(
                {
                    "epoch": epoch,
                    "threshold": validation.threshold,
                    "metrics": asdict(validation.metrics),
                    "selection_value": validation.selection_value,
                }
            )
            if best_validation is None or self.is_better(
                validation.selection_value, best_validation.selection_value
            ):
                model.save(checkpoint_path)
                best_validation = validation

        if best_validation is None:
            message = "Experiment training did not produce validation results"
            self.logger.error(message)
            raise ValueError(message)
        run_dir = self.runs_root / validated_run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        checkpoint_path.replace(run_dir / "model.pkl")
        write_scores(
            run_dir / "validation_scores.csv",
            "validation",
            protocol.validation,
            best_validation.scores,
            best_validation.threshold,
        )
        (run_dir / "validation_history.json").write_text(
            json.dumps(history, indent=2, sort_keys=True) + "\n"
        )
        (run_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "selected_epoch": best_validation.epoch,
                    "threshold": best_validation.threshold,
                    "selection": self.definition["selection"],
                    "selection_value": best_validation.selection_value,
                    "validation": asdict(best_validation.metrics),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        write_manifest(
            run_dir / "manifest.json",
            {
                "artifact_type": "validated",
                "dataset_manifest": dataset_manifest,
                "experiment": self.definition,
                "model": {
                    "module": self.model_module,
                    "class": self.model_type.__name__,
                },
                "protocol": {
                    "module": self.protocol_module,
                    "class": self.protocol_type.__name__,
                },
                "threshold_selector": self.component_identity(
                    self.threshold_module, self.threshold_type
                ),
                "selected_epoch": best_validation.epoch,
                "selected_threshold": best_validation.threshold,
                "selection": self.definition["selection"],
                "selection_value": best_validation.selection_value,
            },
        )
        self.logger.info(
            f"Completed training: {self.definition['selection']['metric']}={best_validation.selection_value}, run_dir={run_dir}"
        )
        return ValidatedRun(run_dir, best_validation)

    def validate(self, model: object, protocol: object, epoch: int) -> ValidationResult:
        """Calibrate a trained model on validation data without fitting it."""

        # .1. score model on validation split
        scores = protocol.score_validation(model)
        metrics_logger = self.logger.for_component("mark_ad_benchmark.metrics.point")
        selector = self.threshold_type(self.logger.for_component(self.threshold_module))
        threshold = selector.fit(scores, protocol.validation.labels)
        metrics = point_metrics(
            protocol.validation.labels, scores, threshold, metrics_logger
        )
        selection_value = getattr(metrics, self.definition["selection"]["metric"])
        self.logger.info(
            f"Validated epoch {epoch}: {self.definition['selection']['metric']}={selection_value}"
        )
        return ValidationResult(epoch, threshold, metrics, selection_value, scores)

    def test(self, validated_run_id: str, run_id: str | None = None) -> TestResult:
        """Evaluate a validated artifact on test data without fitting it."""
        validated_dir = self.runs_root / validated_run_id
        manifest_path = validated_dir / "manifest.json"
        if not manifest_path.is_file():
            message = f"Validated artifact not found: {validated_dir}"
            self.logger.error(message)
            raise FileNotFoundError(message)
        validated_manifest = json.loads(manifest_path.read_text())
        if validated_manifest["artifact_type"] != "validated":
            message = f"Artifact is not a validated run: {validated_dir}"
            self.logger.error(message)
            raise ValueError(message)
        if validated_manifest["experiment"] != self.definition:
            message = "Validated artifact does not match the experiment definition"
            self.logger.error(message)
            raise ValueError(message)
        if validated_manifest["model"] != self.component_identity(
            self.model_module, self.model_type
        ) or validated_manifest["protocol"] != self.component_identity(
            self.protocol_module, self.protocol_type
        ):
            message = (
                "Validated artifact components do not match the available catalogs"
            )
            self.logger.error(message)
            raise ValueError(message)
        if validated_manifest["threshold_selector"] != self.component_identity(
            self.threshold_module, self.threshold_type
        ):
            message = "Validated artifact threshold selector does not match the available catalog"
            self.logger.error(message)
            raise ValueError(message)

        dataset_path, dataset_manifest = self.load_dataset_manifest()
        if validated_manifest["dataset_manifest"] != dataset_manifest:
            message = "Validated artifact does not match the dataset manifest"
            self.logger.error(message)
            raise ValueError(message)
        model = self.model_type.load(
            validated_dir / "model.pkl", self.logger.for_component(self.model_module)
        )
        splits = load_refined_dataset(
            dataset_path,
            self.refined_root,
            self.logger.for_component("mark_ad_benchmark.datasets.loader"),
            self.protocol_type.test_required_splits,
        )
        test_split = splits["test"]
        scores = self.protocol_type.score_test(
            model, test_split, self.logger.for_component(self.protocol_module)
        )
        threshold = validated_manifest["selected_threshold"]
        metrics = point_metrics(
            test_split.labels,
            scores,
            threshold,
            self.logger.for_component("mark_ad_benchmark.metrics.point"),
        )
        test_run_id = run_id or create_run_id(f"{self.definition['id']}-test")
        run_dir = self.runs_root / test_run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        write_scores(run_dir / "test_scores.csv", "test", test_split, scores, threshold)
        (run_dir / "metrics.json").write_text(
            json.dumps(
                {"threshold": threshold, "test": asdict(metrics)},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        write_manifest(
            run_dir / "manifest.json",
            {
                "artifact_type": "test",
                "experiment": self.definition,
                "validated_run": validated_run_id,
                "threshold": threshold,
            },
        )
        self.logger.info(f"Completed testing: test_f1={metrics.f1}, run_dir={run_dir}")
        return TestResult(run_dir, threshold, metrics)

    def run(self, run_id: str | None = None) -> ExperimentResult:
        """Train, validate, and test one experiment definition."""
        validated = self.train(f"{run_id}-validated" if run_id else None)
        test = self.test(
            validated.run_dir.name,
            f"{run_id}-test" if run_id else None,
        )
        return ExperimentResult(validated, test)

    def validate_definition(self) -> None:
        """Reject experiment definitions that cannot run the epoch lifecycle."""
        training = self.definition.get("training")
        if not isinstance(training, dict) or not isinstance(
            training.get("epochs"), int
        ):
            message = "Experiment training.epochs must be an integer"
            self.logger.error(message)
            raise TypeError(message)
        if training["epochs"] < 1:
            message = "Experiment training.epochs must be at least one"
            self.logger.error(message)
            raise ValueError(message)
        threshold = self.definition.get("threshold")
        if not isinstance(threshold, dict) or not isinstance(
            threshold.get("name"), str
        ):
            message = "Experiment threshold.name must be a string"
            self.logger.error(message)
            raise TypeError(message)
        selection = self.definition.get("selection")
        if not isinstance(selection, dict):
            message = "Experiment selection must be a mapping"
            self.logger.error(message)
            raise TypeError(message)
        if selection.get("metric") not in PointMetrics.__dataclass_fields__:
            message = "Experiment selection.metric must be a point metric"
            self.logger.error(message)
            raise ValueError(message)
        if selection.get("direction") not in {"maximize", "minimize"}:
            message = "Experiment selection.direction must be maximize or minimize"
            self.logger.error(message)
            raise ValueError(message)

    def is_better(self, candidate: float, current: float) -> bool:
        """Compare validation values using the configured selection direction."""
        if self.definition["selection"]["direction"] == "maximize":
            return candidate > current
        return candidate < current

    def load_dataset_manifest(self) -> tuple[Path, dict[str, object]]:
        """Load the dataset manifest referenced by this experiment."""
        path = self.manifest_root / f"{self.definition['dataset']}.yaml"
        if not path.is_file():
            message = f"Dataset manifest not found: {path}"
            self.logger.error(message)
            raise FileNotFoundError(message)
        return path, yaml.safe_load(path.read_text())

    @staticmethod
    def component_identity(module: str, component_type: object) -> dict[str, str]:
        return {"module": module, "class": component_type.__name__}


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
