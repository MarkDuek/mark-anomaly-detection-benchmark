---
title: Models
description: Methods evaluated by the benchmark.
---

Models transform multivariate feature rows into higher-is-more-anomalous scores. Their declared lifecycle determines whether they learn from a training split or score a complete batch.

Model constructors accept explicit keyword-only settings and a component logger. The experiment runner unpacks YAML parameter mappings into these arguments; each model keeps one source of truth for its settings and assembles its configuration when saving learned state.

## Available Models

<div class="documentation-map">
  <a href="/models/pca/"><strong>PCA</strong><span>Train-split principal-component reconstruction residuals.</span></a>
  <a href="/models/maxent-pca/"><strong>MaxEnt-PCA</strong><span>Sampled-pair maximum-entropy subspace learning with SGD.</span></a>
  <a href="/models/convolutional-pca/"><strong>ConvPCA</strong><span>Finite temporal filters with per-timestamp reconstruction residuals.</span></a>
  <a href="/models/nystrom-kpca/"><strong>Nyström KPCA</strong><span>Landmark-approximated kernel PCA with feature-space reconstruction residuals.</span></a>
  <a href="/models/robust-pca/"><strong>Robust PCA</strong><span>Batch low-rank plus sparse decomposition with L1 anomaly scores.</span></a>
</div>
