---
title: Multivariate Anomaly Detection
description: A benchmark-oriented workspace for evaluating anomaly-detection methods on multivariate time-series data.
---

<div class="landing-intro">
  <p>A precise, reproducible reference for evaluating anomaly-detection methods on multivariate time-series data.</p>
</div>

## Documentation Map

The benchmark is organized around the components needed to make experiments legible and comparable.

<div class="documentation-map">
  <a href="/models/"><strong>Models</strong><span>Methods evaluated by the benchmark.</span></a>
  <a href="/protocols/"><strong>Protocols</strong><span>Evaluation procedures and data splits.</span></a>
  <a href="/thresholds/"><strong>Thresholds</strong><span>Rules for converting scores into decisions.</span></a>
  <a href="/datasets/"><strong>Datasets</strong><span>Time-series data used for experiments.</span></a>
  <a href="/metrics/"><strong>Metrics</strong><span>Measures used to assess results.</span></a>
  <a href="/runs/"><strong>Runs</strong><span>Experiment records and future visual analysis.</span></a>
</div>

## Notation Preview

Mathematical notation is available throughout the documentation. For example, the familiar F-score can be written as $F_1$:

$$
F_1 = 2 \frac{\mathrm{precision}\,\mathrm{recall}}
{\mathrm{precision}+\mathrm{recall}}
$$

## Reference Styling

| Field | Example | Purpose |
| --- | ---: | --- |
| `run_id` | `calit2-pca-inductive` | Experiment identifier |
| `metric` | $F_1$ | Evaluation label |

```text
experiment: calit2-pca-inductive
```
