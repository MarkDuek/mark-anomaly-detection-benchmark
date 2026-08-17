import math
from collections.abc import Sequence
from dataclasses import dataclass

from mark_ad_benchmark.logging import ComponentLogger


@dataclass(frozen=True)
class PointMetrics:
    precision: float
    recall: float
    f1: float


def point_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float,
    logger: ComponentLogger,
) -> PointMetrics:
    """Calculate point precision, recall, and F1 at a fixed score threshold."""
    validate_inputs(labels, scores, logger)
    metrics = calculate_point_metrics(labels, scores, threshold)
    logger.info(
        f"Point metrics at threshold {threshold}: "
        f"precision={metrics.precision}, recall={metrics.recall}, f1={metrics.f1}"
    )
    return metrics


def validate_inputs(
    labels: Sequence[int], scores: Sequence[float], logger: ComponentLogger
) -> None:
    if len(labels) != len(scores):
        message = "Labels and scores must have equal lengths"
        logger.error(message)
        raise ValueError(message)
    if any(label not in {0, 1} for label in labels):
        message = "Point metrics require binary labels"
        logger.error(message)
        raise ValueError(message)
    if any(not math.isfinite(score) for score in scores):
        message = "Point metrics require finite scores"
        logger.error(message)
        raise ValueError(message)


def calculate_point_metrics(
    labels: Sequence[int], scores: Sequence[float], threshold: float
) -> PointMetrics:
    true_positives = sum(
        label == 1 and score >= threshold for label, score in zip(labels, scores)
    )
    false_positives = sum(
        label == 0 and score >= threshold for label, score in zip(labels, scores)
    )
    false_negatives = sum(
        label == 1 and score < threshold for label, score in zip(labels, scores)
    )
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return PointMetrics(precision, recall, f1)
