import csv
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib import import_module
from itertools import product
from pathlib import Path

import yaml
from tqdm import tqdm

from mark_ad_benchmark.datasets.loader import DatasetSplit, load_refined_dataset
from mark_ad_benchmark.logging import ComponentLogger
from mark_ad_benchmark.metrics import (
    EvaluationMetrics,
    MetricSettings,
    PointMetrics,
    evaluate_metrics,
    point_metrics,
)
from mark_ad_benchmark.models import InductiveAnomalyModel, TransductiveAnomalyModel


@dataclass(frozen=True)
class ValidationResult:
    epoch: int
    threshold: float
    threshold_state: dict[str, object]
    metrics: EvaluationMetrics
    selection_value: float
    scores: tuple[float, ...]
    model_diagnostics: dict[str, object]
    model_parameters: dict[str, object]


@dataclass(frozen=True)
class ValidatedRun:
    run_dir: Path
    validation: ValidationResult


@dataclass(frozen=True)
class TestResult:
    run_dir: Path
    threshold: float
    test_metrics: EvaluationMetrics


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
        self.validate_component_compatibility()

    @classmethod
    def from_path(
        cls, experiment_path: Path, logger: ComponentLogger, **kwargs: Path
    ) -> "Experiment":
        """Create an experiment from a checked-in YAML definition."""
        return cls(yaml.safe_load(experiment_path.read_text()), logger, **kwargs)

    def train(self, run_id: str | None = None) -> ValidatedRun:
        """Train and validate every epoch, then persist the selected model."""
        phase = (
            "TRAINING AND VALIDATION"
            if self.execution_mode == "inductive"
            else "VALIDATION"
        )
        self.logger.info(f"==================== {phase} ====================")
        self.logger.info(
            "Loading training and validation data"
            if self.execution_mode == "inductive"
            else "Loading validation data"
        )
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

        # 3. run set up
        validated_run_id = run_id or create_run_id(f"{self.definition['id']}-validated")
        self.runs_root.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.runs_root / f".{validated_run_id}.checkpoint"
        best_validation: ValidationResult | None = None
        history: list[dict[str, object]] = []

        # 4. validate every configured model parameter candidate
        parameter_candidates = self.model_parameter_candidates()
        epochs = self.definition["training"]["epochs"]
        candidate_label = (
            "candidate" if len(parameter_candidates) == 1 else "candidates"
        )
        epoch_label = "epoch" if epochs == 1 else "epochs"
        self.logger.info(
            f"Validating {len(parameter_candidates)} parameter {candidate_label} across {epochs} {epoch_label}"
        )
        with tqdm(
            total=len(parameter_candidates),
            desc="Validation candidates",
            unit="candidate",
            disable=not sys.stderr.isatty(),
        ) as progress:
            for candidate_index, parameters in enumerate(parameter_candidates, start=1):
                model = self.create_model(parameters)
                for epoch in range(1, epochs + 1):
                    candidate_message = (
                        f"candidate {candidate_index}/{len(parameter_candidates)}, "
                        f"epoch {epoch}/{epochs}: parameters={parameters}"
                    )
                    if self.execution_mode == "inductive":
                        self.logger.info(f"Training {candidate_message}")
                        protocol.fit(model)
                    self.logger.info(f"Validating {candidate_message}")

                    validation = self.validate(model, protocol, epoch, parameters)
                    history.append(
                        {
                            "epoch": epoch,
                            "model_parameters": validation.model_parameters,
                            "threshold": validation.threshold,
                            "threshold_state": validation.threshold_state,
                            "metrics": asdict(validation.metrics),
                            "selection_value": validation.selection_value,
                            "model_diagnostics": validation.model_diagnostics,
                        }
                    )
                    if best_validation is None or self.is_better(
                        validation.selection_value, best_validation.selection_value
                    ):
                        if self.execution_mode == "inductive":
                            model.save(checkpoint_path)
                        best_validation = validation
                progress.update()

        if best_validation is None:
            message = "Experiment training did not produce validation results"
            self.logger.error(message)
            raise ValueError(message)
        run_dir = self.runs_root / validated_run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        if self.execution_mode == "inductive":
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
                    "model_diagnostics": best_validation.model_diagnostics,
                    "selected_model_parameters": best_validation.model_parameters,
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
                    "execution_mode": self.execution_mode,
                },
                "threshold_selector": self.component_identity(
                    self.threshold_module, self.threshold_type
                ),
                "selected_epoch": best_validation.epoch,
                "selected_threshold": best_validation.threshold,
                "threshold_state": best_validation.threshold_state,
                "selected_model_parameters": best_validation.model_parameters,
                "selection": self.definition["selection"],
                "selection_value": best_validation.selection_value,
                "model_diagnostics": best_validation.model_diagnostics,
            },
        )
        self.logger.info(
            f"Completed validation phase: {self.definition['selection']['metric']}={best_validation.selection_value}, parameters={best_validation.model_parameters}, run_dir={run_dir}"
        )
        return ValidatedRun(run_dir, best_validation)

    def validate(
        self,
        model: object,
        protocol: object,
        epoch: int,
        model_parameters: dict[str, object],
    ) -> ValidationResult:
        """Calibrate one protocol-compatible model on validation data."""

        if self.execution_mode == "inductive":
            scores = protocol.score_validation(model)
        else:
            scores = protocol.fit_score_validation(model)
        metrics_logger = self.logger.for_component("mark_ad_benchmark.metrics.point")
        selector = self.threshold_type(self.logger.for_component(self.threshold_module))
        threshold = selector.fit(scores, protocol.validation.labels)
        point = point_metrics(
            protocol.validation.labels, scores, threshold, metrics_logger
        )
        metrics = evaluate_metrics(
            protocol.validation.labels,
            scores,
            threshold,
            self.metric_settings(),
            self.logger.for_component("mark_ad_benchmark.metrics.evaluation"),
        )
        selection_value = getattr(point, self.definition["selection"]["metric"])
        self.logger.info(
            f"Validated epoch {epoch}: {self.definition['selection']['metric']}={selection_value}"
        )
        return ValidationResult(
            epoch,
            threshold,
            selector.state(),
            metrics,
            selection_value,
            scores,
            self.model_diagnostics(model),
            model_parameters,
        )

    def test(self, validated_run_id: str, run_id: str | None = None) -> TestResult:
        """Evaluate a validated artifact using its protocol's test lifecycle."""
        self.logger.info("========================= TEST =========================")
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
        if (
            validated_manifest["model"]
            != self.component_identity(self.model_module, self.model_type)
            or validated_manifest["protocol"] != self.protocol_identity()
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

        self.logger.info("Loading test data")
        dataset_path, dataset_manifest = self.load_dataset_manifest()
        if validated_manifest["dataset_manifest"] != dataset_manifest:
            message = "Validated artifact does not match the dataset manifest"
            self.logger.error(message)
            raise ValueError(message)
        if self.execution_mode == "inductive":
            model = self.model_type.load(
                validated_dir / "model.pkl",
                self.logger.for_component(self.model_module),
            )
        else:
            model = self.create_model(validated_manifest["selected_model_parameters"])
        splits = load_refined_dataset(
            dataset_path,
            self.refined_root,
            self.logger.for_component("mark_ad_benchmark.datasets.loader"),
            self.protocol_type.test_required_splits,
        )
        test_split = splits["test"]
        self.logger.info(
            "Scoring test data with selected validation configuration: "
            f"parameters={validated_manifest['selected_model_parameters']}"
        )
        if self.execution_mode == "inductive":
            scores = self.protocol_type.score_test(
                model, test_split, self.logger.for_component(self.protocol_module)
            )
        else:
            scores = self.protocol_type.fit_score_test(
                model, test_split, self.logger.for_component(self.protocol_module)
            )
        selector = self.threshold_type(self.logger.for_component(self.threshold_module))
        threshold_state = validated_manifest.get(
            "threshold_state",
            {"threshold": validated_manifest["selected_threshold"]},
        )
        if not isinstance(threshold_state, dict):
            message = "Validated artifact threshold state must be a mapping"
            self.logger.error(message)
            raise TypeError(message)
        selector.load_state(threshold_state)
        threshold = selector.threshold_for(scores)
        self.logger.info(f"Applied threshold to test scores: {threshold}")
        point_metrics(
            test_split.labels,
            scores,
            threshold,
            self.logger.for_component("mark_ad_benchmark.metrics.point"),
        )
        metrics = evaluate_metrics(
            test_split.labels,
            scores,
            threshold,
            self.metric_settings(),
            self.logger.for_component("mark_ad_benchmark.metrics.evaluation"),
        )
        test_run_id = run_id or create_run_id(f"{self.definition['id']}-test")
        run_dir = self.runs_root / test_run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        write_scores(run_dir / "test_scores.csv", "test", test_split, scores, threshold)
        (run_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "threshold": threshold,
                    "test": asdict(metrics),
                    "model_diagnostics": self.model_diagnostics(model),
                },
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
                "model_diagnostics": self.model_diagnostics(model),
                "selected_model_parameters": validated_manifest[
                    "selected_model_parameters"
                ],
            },
        )
        self.logger.info(
            f"Completed test phase: test_f1={metrics.f1}, run_dir={run_dir}"
        )
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
        model = self.definition.get("model")
        if not isinstance(model, dict) or not isinstance(model.get("parameters"), dict):
            message = "Experiment model.parameters must be a mapping"
            self.logger.error(message)
            raise TypeError(message)
        candidates = model.get("candidates", {})
        if isinstance(candidates, list):
            valid_candidates = bool(candidates) and all(
                isinstance(candidate, dict)
                and all(isinstance(name, str) for name in candidate)
                for candidate in candidates
            )
        else:
            valid_candidates = isinstance(candidates, dict) and all(
                isinstance(name, str) and isinstance(values, list) and bool(values)
                for name, values in candidates.items()
            )
        if not valid_candidates:
            message = "Experiment model.candidates must map names to nonempty lists or be a nonempty list of parameter mappings"
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
        self.metric_settings()

    def is_better(self, candidate: float, current: float) -> bool:
        """Compare validation values using the configured selection direction."""
        if self.definition["selection"]["direction"] == "maximize":
            return candidate > current
        return candidate < current

    @property
    def execution_mode(self) -> str:
        """Return the lifecycle mode declared by the selected protocol."""
        return self.protocol_type.execution_mode

    def validate_component_compatibility(self) -> None:
        """Reject model and protocol combinations with incompatible lifecycles."""
        mode = getattr(self.protocol_type, "execution_mode", None)
        if mode == "inductive":
            compatible = issubclass(self.model_type, InductiveAnomalyModel)
        elif mode == "transductive":
            compatible = issubclass(self.model_type, TransductiveAnomalyModel)
            if self.definition["training"]["epochs"] != 1:
                message = (
                    "Transductive experiments require training.epochs to equal one"
                )
                self.logger.error(message)
                raise ValueError(message)
        else:
            message = f"Unsupported protocol execution mode: {mode}"
            self.logger.error(message)
            raise ValueError(message)
        if not compatible:
            message = f"Model is incompatible with {mode} protocol"
            self.logger.error(message)
            raise TypeError(message)

    def create_model(self, parameters: dict[str, object] | None = None) -> object:
        """Instantiate the configured model with a component-scoped logger."""
        return self.model_type(
            parameters or self.definition["model"]["parameters"],
            self.logger.for_component(self.model_module),
        )

    def model_parameter_candidates(self) -> tuple[dict[str, object], ...]:
        """Expand a parameter grid or merge explicit candidates with base parameters."""
        model = self.definition["model"]
        parameters = model["parameters"]
        candidates = model.get("candidates", {})
        if not candidates:
            return (dict(parameters),)
        if isinstance(candidates, list):
            return tuple({**parameters, **candidate} for candidate in candidates)
        names = tuple(candidates)
        return tuple(
            {**parameters, **dict(zip(names, values, strict=True))}
            for values in product(*(candidates[name] for name in names))
        )

    def metric_settings(self) -> MetricSettings:
        """Read explicit range-metric settings from the experiment definition."""
        metrics = self.definition.get("metrics")
        if not isinstance(metrics, dict):
            message = "Experiment metrics must be a mapping"
            self.logger.error(message)
            raise TypeError(message)
        vus = metrics.get("vus")
        range_settings = metrics.get("range")
        if not isinstance(vus, dict) or not isinstance(range_settings, dict):
            message = "Experiment metrics.vus and metrics.range must be mappings"
            self.logger.error(message)
            raise TypeError(message)
        max_window = vus.get("max_window")
        samples = vus.get("threshold_samples")
        existence_weight = range_settings.get("recall_existence_weight")
        if not isinstance(max_window, int) or max_window < 0:
            message = "Experiment metrics.vus.max_window must be a nonnegative integer"
            self.logger.error(message)
            raise ValueError(message)
        if not isinstance(samples, int) or samples < 1:
            message = (
                "Experiment metrics.vus.threshold_samples must be a positive integer"
            )
            self.logger.error(message)
            raise ValueError(message)
        if (
            not isinstance(existence_weight, (int, float))
            or isinstance(existence_weight, bool)
            or not 0 <= existence_weight <= 1
        ):
            message = "Experiment metrics.range.recall_existence_weight must be between zero and one"
            self.logger.error(message)
            raise ValueError(message)
        return MetricSettings(max_window, samples, float(existence_weight))

    def model_diagnostics(self, model: object) -> dict[str, object]:
        """Return diagnostics exposed by the selected model."""
        return dict(getattr(model, "diagnostics", {}))

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

    def protocol_identity(self) -> dict[str, str]:
        """Return the protocol identity including its declared lifecycle mode."""
        return {
            **self.component_identity(self.protocol_module, self.protocol_type),
            "execution_mode": self.execution_mode,
        }


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
