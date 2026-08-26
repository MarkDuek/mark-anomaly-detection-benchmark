import csv
import html
import json
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
) -> Path:
    """Write a browser dashboard for every saved test-run artifact."""
    test_run_ids = sorted(
        run_dir.name for run_dir in runs_root.iterdir() if is_test_run(run_dir)
    )
    if not test_run_ids:
        message = f"No test artifacts found in {runs_root}"
        logger.error(message)
        raise FileNotFoundError(message)
    datasets: dict[str, tuple[DatasetSplit, list[str]]] = {}
    runs = {
        test_run_id: load_run_data(
            test_run_id,
            logger,
            manifest_root,
            refined_root,
            runs_root,
            datasets,
        )
        for test_run_id in test_run_ids
    }
    output_path = runs_root / "visualization.html"
    output_path.write_text(dashboard_html(runs), encoding="utf-8")
    logger.info(f"Wrote visualization dashboard: {output_path}")
    return output_path


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
    datasets: dict[str, tuple[DatasetSplit, list[str]]],
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
        )
    test_split, feature_names = datasets[dataset_id]
    scores = load_scores(test_run_dir / "test_scores.csv", test_split, logger)
    metrics = json.loads((test_run_dir / "metrics.json").read_text())
    test_metrics = metrics["test"]
    indexes = display_indexes(test_split.labels)
    return {
        "experiment_id": test_manifest["experiment"]["id"],
        "timestamps": [test_split.timestamps[index] for index in indexes],
        "labels": [test_split.labels[index] for index in indexes],
        "scores": [scores[index] for index in indexes],
        "threshold": metrics["threshold"],
        "metrics": test_metrics.get("point", test_metrics),
        "metric_groups": test_metrics,
        "features": {
            feature_name: [
                test_split.features[index][feature_index] for index in indexes
            ]
            for feature_index, feature_name in enumerate(feature_names)
        },
    }


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


def dashboard_html(runs: dict[str, dict[str, object]]) -> str:
    options = "\n".join(
        f'<option value="{html.escape(run_id)}">{html.escape(data["experiment_id"])} ({html.escape(run_id)})</option>'
        for run_id, data in runs.items()
    )
    run_json = json.dumps(runs, cls=PlotlyJSONEncoder)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Anomaly Detection Results</title>
  <script src="https://cdn.plot.ly/plotly-3.4.0.min.js"></script>
  <style>
    body {{ margin: 0 auto; max-width: 1440px; padding: 24px; font-family: Arial, sans-serif; color: #172033; }}
    h1, h2 {{ margin: 0 0 12px; }}
    label {{ display: block; font-weight: 700; margin: 12px 0 4px; }}
    select {{ min-width: 360px; padding: 6px; }}
    #comparison-runs {{ display: flex; flex-wrap: wrap; gap: 8px 16px; margin: 8px 0 20px; }}
    #comparison-runs label {{ display: flex; align-items: center; gap: 6px; margin: 0; font-weight: 400; }}
    #metric-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 12px 0 24px; }}
    .metric-card {{ border: 1px solid #d8deea; border-radius: 8px; padding: 12px; background: #f8fafc; }}
    .metric-card-name {{ color: #526174; font-size: 12px; text-transform: uppercase; }}
    .metric-card-value {{ font-size: 24px; font-weight: 700; margin-top: 4px; }}
    #comparison-table {{ border-collapse: collapse; margin: 12px 0 24px; max-width: 100%; overflow-x: auto; display: block; }}
    #comparison-table th, #comparison-table td {{ border: 1px solid #d8deea; padding: 8px; text-align: right; white-space: nowrap; }}
    #comparison-table th:first-child, #comparison-table td:first-child {{ text-align: left; }}
    #comparison-table th {{ background: #eaf0f8; }}
    @media (max-width: 600px) {{ body {{ padding: 16px; }} select {{ min-width: 100%; }} }}
  </style>
</head>
<body>
  <h1>Anomaly Detection Results</h1>
  <label for="run">Test run</label>
  <select id="run">{options}</select>
  <label for="features">Features</label>
  <select id="features" multiple size="8"></select>
  <h2>Metrics</h2>
  <div id="metric-cards"></div>
  <h2>Compare runs</h2>
  <div id="comparison-runs"></div>
  <table id="comparison-table"></table>
  <div id="features-chart"></div>
  <div id="scores-chart"></div>
  <div id="predictions-chart"></div>
  <script>
    const runs = {run_json};
    const runSelector = document.getElementById("run");
    const featureSelector = document.getElementById("features");
    const comparisonRuns = document.getElementById("comparison-runs");
    function selectedFeatures() {{
      return Array.from(featureSelector.selectedOptions).map(option => option.value);
    }}
    function metricEntries(run) {{
      return Object.entries(run.metric_groups).flatMap(([group, metrics]) =>
        typeof metrics === "object" && metrics !== null
          ? Object.entries(metrics).map(([name, value]) => [
              `${{group.replaceAll("_", " ")}} ${{name.replaceAll("_", " ")}}`, value
            ])
          : [[group.replaceAll("_", " "), metrics]]
      );
    }}
    function formatMetric(value) {{
      return value === null || value === undefined ? "N/A" : Number(value).toFixed(4);
    }}
    function renderMetricCards(run) {{
      const cards = document.getElementById("metric-cards");
      cards.replaceChildren(...metricEntries(run).map(([name, value]) => {{
        const card = document.createElement("div");
        card.className = "metric-card";
        const label = document.createElement("div");
        label.className = "metric-card-name";
        label.textContent = name;
        const metric = document.createElement("div");
        metric.className = "metric-card-value";
        metric.textContent = formatMetric(value);
        card.append(label, metric);
        return card;
      }}));
    }}
    function selectedComparisonRuns() {{
      return Array.from(comparisonRuns.querySelectorAll("input:checked")).map(input => input.value);
    }}
    function renderComparison() {{
      const runIds = selectedComparisonRuns();
      const table = document.getElementById("comparison-table");
      const metricNames = [...new Set(runIds.flatMap(runId => metricEntries(runs[runId]).map(([name]) => name)))];
      table.replaceChildren();
      if (!runIds.length) {{
        table.createCaption().textContent = "Select runs to compare their metrics.";
        return;
      }}
      const header = document.createElement("tr");
      header.append(Object.assign(document.createElement("th"), {{textContent: "Metric"}}));
      for (const runId of runIds) {{
        header.append(Object.assign(document.createElement("th"), {{textContent: runs[runId].experiment_id}}));
      }}
      table.append(header);
      for (const metricName of metricNames) {{
        const row = document.createElement("tr");
        row.append(Object.assign(document.createElement("td"), {{textContent: metricName}}));
        for (const runId of runIds) {{
          const values = new Map(metricEntries(runs[runId]));
          row.append(Object.assign(document.createElement("td"), {{textContent: formatMetric(values.get(metricName))}}));
        }}
        table.append(row);
      }}
    }}
    function populateComparisonRuns() {{
      comparisonRuns.replaceChildren(...Object.entries(runs).map(([runId, run]) => {{
        const label = document.createElement("label");
        const input = document.createElement("input");
        input.type = "checkbox";
        input.value = runId;
        input.checked = runId === runSelector.value;
        input.addEventListener("change", renderComparison);
        label.append(input, document.createTextNode(run.experiment_id));
        return label;
      }}));
    }}
    function populateFeatures() {{
      const features = Object.keys(runs[runSelector.value].features);
      featureSelector.replaceChildren(...features.map((feature, index) => {{
        const option = document.createElement("option");
        option.value = feature;
        option.textContent = feature;
        option.selected = index === 0;
        return option;
      }}));
    }}
    function render() {{
      const run = runs[runSelector.value];
      const anomalies = run.labels.map((label, index) => label ? index : -1).filter(index => index >= 0);
      const features = selectedFeatures();
      const featureTraces = features.flatMap(feature => [
        {{x: run.timestamps, y: run.features[feature], mode: "lines", name: feature}},
        {{x: anomalies.map(index => run.timestamps[index]), y: anomalies.map(index => run.features[feature][index]), mode: "markers", marker: {{color: "red", size: 7}}, name: "True anomaly", showlegend: false}}
      ]);
      Plotly.react("features-chart", featureTraces, {{title: "Selected features", hovermode: "x unified"}});
      Plotly.react("scores-chart", [
        {{x: run.timestamps, y: run.scores, mode: "lines", name: "Anomaly score"}},
        {{x: anomalies.map(index => run.timestamps[index]), y: anomalies.map(index => run.scores[index]), mode: "markers", marker: {{color: "red", size: 7}}, name: "True anomaly"}}
      ], {{title: "Anomaly score", hovermode: "x unified", shapes: [{{type: "line", xref: "paper", x0: 0, x1: 1, y0: run.threshold, y1: run.threshold, line: {{dash: "dash", color: "orange"}}}}]}});
      const predictions = run.scores.map(score => Number(score >= run.threshold));
      Plotly.react("predictions-chart", [
        {{x: run.timestamps, y: predictions, mode: "lines", line: {{shape: "hv"}}, name: "Prediction"}},
        {{x: anomalies.map(index => run.timestamps[index]), y: anomalies.map(() => 1), mode: "markers", marker: {{color: "red", size: 7}}, name: "True anomaly"}}
      ], {{title: "Prediction", hovermode: "x unified", yaxis: {{range: [-0.1, 1.1]}}}});
      renderMetricCards(run);
    }}
    runSelector.addEventListener("change", () => {{ populateFeatures(); populateComparisonRuns(); render(); renderComparison(); }});
    featureSelector.addEventListener("change", render);
    populateFeatures();
    populateComparisonRuns();
    render();
    renderComparison();
  </script>
</body>
</html>
"""
