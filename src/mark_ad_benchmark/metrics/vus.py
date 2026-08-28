import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from vus.analysis.robustness_eval import generate_curve

from mark_ad_benchmark.metrics.events import ranges


@dataclass(frozen=True)
class VusRankingMetrics:
    auc_roc: float | None
    auc_pr: float | None


@dataclass(frozen=True)
class VusPredictionMetrics:
    precision: float
    recall: float
    f1: float


def vus_ranking_metrics(
    labels: Sequence[int], scores: Sequence[float], max_window: int, samples: int
) -> VusRankingMetrics:
    """Calculate canonical VUS-ROC and VUS-PR metrics."""
    if not any(labels) or all(labels):
        return VusRankingMetrics(auc_roc=None, auc_pr=None)
    *_, auc_roc, auc_pr = generate_curve(
        np.asarray(labels),
        np.asarray(scores),
        slidingWindow=max_window,
        version="opt",
        thre=samples,
    )
    return VusRankingMetrics(float(auc_roc), float(auc_pr))


def vus_prediction_metrics(
    labels: Sequence[int],
    predictions: Sequence[int],
    max_window: int,
    progress: Callable[[], None] | None = None,
) -> VusPredictionMetrics:
    """Average TSB-AD range precision, recall, and F1 over extension windows."""
    if not any(labels) or not any(predictions):
        if progress is not None:
            for _ in range(max_window + 1):
                progress()
        return VusPredictionMetrics(precision=0.0, recall=0.0, f1=0.0)
    values = []
    for window in range(max_window + 1):
        values.append(
            range_counts(labels, predictions, window, include_existence=False)
        )
        if progress is not None:
            progress()
    precision = sum(value[0] for value in values) / len(values)
    recall = sum(value[1] for value in values) / len(values)
    f1 = sum(value[2] for value in values) / len(values)
    return VusPredictionMetrics(precision, recall, f1)


def range_counts(
    labels: Sequence[int],
    predictions: Sequence[int],
    window: int,
    include_existence: bool,
) -> tuple[float, float, float, float]:
    """Return soft range precision, recall, F1, and false-positive rate."""
    extended = extend_labels(labels, window)
    prediction = np.asarray(predictions, dtype=float)
    positives = sum(labels)
    product = extended * prediction
    true_positive = float(product.sum())
    predicted = float(prediction.sum())
    range_list = influence_ranges(labels, window)
    hit_ratio = (
        sum(any(predictions[start : end + 1]) for start, end in range_list)
        / len(range_list)
        if range_list
        else 0.0
    )
    new_labels = product.copy()
    new_labels[extended == 1] = 1
    adjusted_positive = (positives + float(new_labels.sum())) / 2
    recall = min(true_positive / adjusted_positive, 1.0) if adjusted_positive else 0.0
    if include_existence:
        recall *= hit_ratio
    precision = true_positive / predicted if predicted else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    false_positive = predicted - true_positive
    negatives = len(labels) - adjusted_positive
    fpr = false_positive / negatives if negatives else 0.0
    return precision, recall, f1, fpr


def extend_labels(labels: Sequence[int], window: int) -> np.ndarray:
    """Apply the TSB-AD square-root range extension for one window size."""
    extended = np.asarray(labels, dtype=np.float32)
    if window == 0:
        return extended
    for start, end in ranges(labels):
        for index in range(max(start - window // 2, 0), start):
            extended[index] += math.sqrt(1 - (start - index) / window)
        for index in range(end + 1, min(end + window // 2 + 1, len(labels))):
            extended[index] += math.sqrt(1 - (index - end) / window)
    return np.minimum(extended, 1.0)


def influence_ranges(labels: Sequence[int], window: int) -> tuple[tuple[int, int], ...]:
    """Return merged range-extension zones used for TSB-AD existence rewards."""
    zones = []
    for start, end in ranges(labels):
        start = max(start - window // 2, 0)
        end = min(end + window // 2, len(labels) - 1)
        if zones and zones[-1][1] >= start:
            zones[-1] = (zones[-1][0], end)
        else:
            zones.append((start, end))
    return tuple(zones)
