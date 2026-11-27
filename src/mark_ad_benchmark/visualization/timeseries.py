import csv
import json
import math
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

import yaml
from plotly.utils import PlotlyJSONEncoder

from mark_ad_benchmark.datasets.loader import DatasetSplit, load_refined_dataset
from mark_ad_benchmark.logging import ComponentLogger
from mark_ad_benchmark.visualization.training import visualize_training

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
    curve_assets: dict[str, dict[str, object]] = {}
    training_assets: dict[str, Path | None] = {}
    for index, test_run_id in enumerate(test_run_ids, start=1):
        logger.info(f"Exporting test run {index}/{len(test_run_ids)}: {test_run_id}")
        test_dir = runs_root / test_run_id
        test_manifest = json.loads((test_dir / "manifest.json").read_text())
        validated_run = test_manifest["validated_run"]
        if validated_run not in training_assets:
            training_assets[validated_run] = visualize_training(
                runs_root / validated_run,
                logger.for_component("mark_ad_benchmark.visualization.training"),
            )
        curve_assets[test_run_id] = {
            "validation": load_run_curves(
                runs_root / test_manifest["validated_run"], "validation", logger
            ),
            "test": load_run_curves(test_dir, "test", logger),
        }
        runs[test_run_id] = load_run_data(
            test_run_id,
            logger,
            manifest_root,
            refined_root,
            runs_root,
            datasets,
        )
        runs[test_run_id]["curves_url"] = f"run-curves/{test_run_id}.json"
        if training_assets[validated_run] is not None:
            runs[test_run_id]["training_url"] = f"run-training/{validated_run}.html"
    for validated_run, path in training_assets.items():
        if path is not None:
            training_dir = docs_output_path.parent / "run-training"
            training_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, training_dir / f"{validated_run}.html")
    curves_dir = docs_output_path.parent / "run-curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    for test_run_id, curves in curve_assets.items():
        (curves_dir / f"{test_run_id}.json").write_text(
            json.dumps(curves, allow_nan=False), encoding="utf-8"
        )
    docs_output_path.parent.mkdir(parents=True, exist_ok=True)
    docs_output_path.write_text(
        json.dumps(runs, cls=PlotlyJSONEncoder), encoding="utf-8"
    )
    logger.info(
        f"Wrote documentation run data for {len(runs)} test {run_label}: {docs_output_path}"
    )
    return docs_output_path


def publish_runs(
    logger: ComponentLogger,
    published_runs_path: Path = Path("docs/published-runs.txt"),
    source_path: Path = Path("docs/public/runs.json"),
    output_path: Path = Path("docs/public/published-runs"),
) -> Path:
    """Package selected existing visualization assets for the documentation site."""
    run_ids = tuple(
        line
        for line in (
            line.strip()
            for line in published_runs_path.read_text(encoding="utf-8").splitlines()
        )
        if line and not line.startswith("#")
    )
    if len(run_ids) != len(set(run_ids)):
        message = f"Duplicate run IDs in {published_runs_path}"
        logger.error(message)
        raise ValueError(message)
    if run_ids and not source_path.is_file():
        message = (
            f"Visualization data not found: {source_path}; run make visualize first"
        )
        logger.error(message)
        raise FileNotFoundError(message)
    source_runs = json.loads(source_path.read_text(encoding="utf-8")) if run_ids else {}
    missing_ids = [run_id for run_id in run_ids if run_id not in source_runs]
    if missing_ids:
        message = f"Published run not found in {source_path}: {missing_ids[0]}"
        logger.error(message)
        raise FileNotFoundError(message)
    selected_runs = {run_id: source_runs[run_id] for run_id in run_ids}
    source_root = source_path.parent
    assets = [
        source_root / asset_path
        for run in selected_runs.values()
        for asset_path in (
            run["curves_url"],
            *([run["training_url"]] if "training_url" in run else []),
        )
    ]
    missing_assets = [path for path in assets if not path.is_file()]
    if missing_assets:
        message = f"Visualization asset not found: {missing_assets[0]}; run make visualize first"
        logger.error(message)
        raise FileNotFoundError(message)
    temporary_path = output_path.with_name(f"{output_path.name}.tmp")
    shutil.rmtree(temporary_path, ignore_errors=True)
    temporary_path.mkdir(parents=True)
    (temporary_path / "runs.json").write_text(
        json.dumps(selected_runs, cls=PlotlyJSONEncoder), encoding="utf-8"
    )
    for asset in assets:
        destination = temporary_path / asset.relative_to(source_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(asset, destination)
    shutil.rmtree(output_path, ignore_errors=True)
    temporary_path.replace(output_path)
    run_label = "run" if len(run_ids) == 1 else "runs"
    logger.info(f"Published {len(run_ids)} {run_label}: {output_path}")
    return output_path


def load_run_curves(
    run_dir: Path, split: str, logger: ComponentLogger
) -> dict[str, object]:
    path = run_dir / "ranking_curves.json"
    if not path.is_file():
        message = f"Ranking curves not found: {path}; regenerate this run"
        logger.error(message)
        raise FileNotFoundError(message)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    threshold = metrics["threshold"]
    curves = json.loads(path.read_text())
    for name, axis in (("roc", "tpr"), ("pr", "recall")):
        curve = curves[name]
        if curve is None:
            continue
        values = curve[axis]
        indexes = [
            index
            for index in range(len(values))
            if index == 0
            or index == len(values) - 1
            or values[index] != values[index - 1]
            or values[index] != values[index + 1]
        ]
        if len(indexes) > MAX_DISPLAY_POINTS:
            indexes = [
                indexes[index * (len(indexes) - 1) // (MAX_DISPLAY_POINTS - 1)]
                for index in range(MAX_DISPLAY_POINTS)
            ]
        curves[name] = {
            key: [points[index] for index in indexes] for key, points in curve.items()
        }
    return {
        "curves": curves,
        "ranking": metrics[split]["ranking"],
        "threshold": threshold if math.isfinite(threshold) else None,
    }


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
        {"pca": "PCA", "robust": "Robust", "maxent": "MaxEnt"}.get(word, word.title())
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
