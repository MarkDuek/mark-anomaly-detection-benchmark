import csv
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import yaml
from plotly.utils import PlotlyJSONEncoder

from mark_ad_benchmark.datasets.loader import DatasetSplit, load_refined_dataset
from mark_ad_benchmark.logging import ComponentLogger

MAX_DISPLAY_POINTS = 5_000


def visualize_runs(
    logger: ComponentLogger,
    manifest_root: Path = Path("data/manifests"),
    refined_root: Path = Path("data/refined"),
    runs_root: Path = Path("runs"),
    docs_output_path: Path = Path("docs/public/runs.json"),
) -> Path:
    """Export every saved test-run artifact for the documentation explorer."""
    test_run_ids = sorted(
        run_dir.name for run_dir in runs_root.iterdir() if is_test_run(run_dir)
    )
    if not test_run_ids:
        message = f"No test artifacts found in {runs_root}"
        logger.error(message)
        raise FileNotFoundError(message)
    datasets: dict[str, tuple[DatasetSplit, list[str], list[str]]] = {}
    run_label = "run" if len(test_run_ids) == 1 else "runs"
    logger.info(f"Exporting {len(test_run_ids)} saved test {run_label}")
    runs: dict[str, dict[str, object]] = {}
    for index, test_run_id in enumerate(test_run_ids, start=1):
        logger.info(f"Exporting test run {index}/{len(test_run_ids)}: {test_run_id}")
        runs[test_run_id] = load_run_data(
            test_run_id,
            logger,
            manifest_root,
            refined_root,
            runs_root,
            datasets,
        )
    docs_output_path.parent.mkdir(parents=True, exist_ok=True)
    docs_output_path.write_text(
        json.dumps(runs, cls=PlotlyJSONEncoder), encoding="utf-8"
    )
    logger.info(
        f"Wrote documentation run data for {len(runs)} test {run_label}: {docs_output_path}"
    )
    return docs_output_path


def is_test_run(run_dir: Path) -> bool:
    manifest_path = run_dir / "manifest.json"
    return (
        manifest_path.is_file()
        and json.loads(manifest_path.read_text())["artifact_type"] == "test"
    )


def display_indexes(labels: tuple[int, ...]) -> tuple[int, ...]:
    """Keep an even display sample while retaining every true anomaly."""
    if len(labels) <= MAX_DISPLAY_POINTS:
        return tuple(range(len(labels)))
    evenly_spaced = {
        index * (len(labels) - 1) // (MAX_DISPLAY_POINTS - 1)
        for index in range(MAX_DISPLAY_POINTS)
    }
    anomalies = {index for index, label in enumerate(labels) if label == 1}
    return tuple(sorted(evenly_spaced | anomalies))


def load_run_data(
    test_run_id: str,
    logger: ComponentLogger,
    manifest_root: Path,
    refined_root: Path,
    runs_root: Path,
    datasets: dict[str, tuple[DatasetSplit, list[str], list[str]]],
) -> dict[str, object]:
    test_run_dir = runs_root / test_run_id
    test_manifest = load_artifact_manifest(test_run_dir, "test", logger)
    validated_manifest = load_artifact_manifest(
        runs_root / test_manifest["validated_run"], "validated", logger
    )
    dataset_id = validated_manifest["experiment"]["dataset"]
    dataset_manifest_path = manifest_root / f"{dataset_id}.yaml"
    if not dataset_manifest_path.is_file():
        message = f"Dataset manifest not found: {dataset_manifest_path}"
        logger.error(message)
        raise FileNotFoundError(message)
    if dataset_id not in datasets:
        dataset_manifest = yaml.safe_load(dataset_manifest_path.read_text())
        datasets[dataset_id] = (
            load_refined_dataset(
                dataset_manifest_path,
                refined_root,
                logger.for_component("mark_ad_benchmark.datasets.loader"),
                ("test",),
            )["test"],
            dataset_manifest["refined_schema"]["features"],
            dataset_manifest["raw_schema"]["features"],
        )
    test_split, feature_names, raw_feature_names = datasets[dataset_id]
    scores = load_scores(test_run_dir / "test_scores.csv", test_split, logger)
    metrics = json.loads((test_run_dir / "metrics.json").read_text())
    test_metrics = metrics["test"]
    experiment = test_manifest["experiment"]
    model = experiment.get("model", {})
    indexes = display_indexes(test_split.labels)
    return {
        "experiment_id": experiment["id"],
        "dataset_id": dataset_id,
        "display_name": run_display_name(test_run_id),
        "run_time": run_time(test_run_id),
        "feature_labels": dict(zip(feature_names, raw_feature_names, strict=True)),
        "timestamps": [test_split.timestamps[index] for index in indexes],
        "labels": [test_split.labels[index] for index in indexes],
        "scores": [scores[index] for index in indexes],
        "threshold": metrics["threshold"],
        "metrics": test_metrics.get("point", test_metrics),
        "metric_groups": test_metrics,
        "configuration": {
            "model": model.get("name"),
            "configured_parameters": model.get("parameters", {}),
            "candidates": model.get("candidates", {}),
            "selected_parameters": test_manifest.get("selected_model_parameters", {}),
            "protocol": experiment.get("protocol"),
            "threshold_selector": experiment.get("threshold", {}).get("name"),
            "validation_threshold": validated_manifest.get("selected_threshold"),
            "threshold_calibration": validated_manifest.get("threshold_state", {}),
            "selection": experiment.get("selection", {}),
            "model_diagnostics": metrics.get("model_diagnostics", {}),
        },
        "features": {
            feature_name: [
                test_split.features[index][feature_index] for index in indexes
            ]
            for feature_index, feature_name in enumerate(feature_names)
        },
    }


def run_display_name(run_id: str) -> str:
    """Format the experiment segment of a versioned run ID for display."""
    experiment = re.sub(r"^\d{8}T\d{6}\d*Z-", "", run_id)
    experiment = re.sub(r"-(validated|test)$", "", experiment)
    return " ".join(
        {"pca": "PCA", "robust": "Robust"}.get(word, word.title())
        for word in experiment.replace("_", "-").split("-")
    )


def run_time(run_id: str) -> str | None:
    """Extract the UTC creation time from a versioned run ID when present."""
    match = re.match(r"^(\d{8}T\d{6})\d*Z-", run_id)
    if match is None:
        return None
    return (
        datetime.strptime(match.group(1), "%Y%m%dT%H%M%S")
        .replace(tzinfo=UTC)
        .strftime("%d %b %Y, %H:%M UTC")
    )


def load_artifact_manifest(
    run_dir: Path, artifact_type: str, logger: ComponentLogger
) -> dict[str, object]:
    path = run_dir / "manifest.json"
    if not path.is_file():
        message = f"{artifact_type.title()} artifact not found: {run_dir}"
        logger.error(message)
        raise FileNotFoundError(message)
    manifest = json.loads(path.read_text())
    if manifest["artifact_type"] != artifact_type:
        message = f"Artifact is not a {artifact_type} run: {run_dir}"
        logger.error(message)
        raise ValueError(message)
    return manifest


def load_scores(
    path: Path, split: DatasetSplit, logger: ComponentLogger
) -> tuple[float, ...]:
    if not path.is_file():
        message = f"Test scores not found: {path}"
        logger.error(message)
        raise FileNotFoundError(message)
    with path.open(newline="", encoding="utf-8") as file:
        rows = tuple(csv.DictReader(file))
    if len(rows) != len(split.timestamps) or any(
        row["timestamp"] != timestamp or int(row["label"]) != label
        for row, timestamp, label in zip(
            rows, split.timestamps, split.labels, strict=True
        )
    ):
        message = "Test scores do not match the refined test split"
        logger.error(message)
        raise ValueError(message)
    return tuple(float(row["score"]) for row in rows)
