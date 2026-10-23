import math
from collections.abc import Sequence
from dataclasses import dataclass

from mark_ad_benchmark.logging import ComponentLogger


@dataclass(frozen=True)
class PointMetrics:
    precision: float
    recall: float
    f1: float


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


def point_metrics(
    labels: Sequence[int], predictions: Sequence[int], logger: ComponentLogger
) -> PointMetrics:
    """Calculate point precision, recall, and F1 from binary predictions."""
    # 1. count confusion-matrix values (documentation excerpt; keep synchronized)
    true_positives = sum(
        label == 1 and prediction == 1 for label, prediction in zip(labels, predictions)
    )
    false_positives = sum(
        label == 0 and prediction == 1 for label, prediction in zip(labels, predictions)
    )
    false_negatives = sum(
        label == 1 and prediction == 0 for label, prediction in zip(labels, predictions)
    )

    # 2. calculate precision (documentation excerpt; keep synchronized)
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )

    # 3. calculate recall (documentation excerpt; keep synchronized)
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )

    # 4. calculate F1 (documentation excerpt; keep synchronized)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    logger.info(f"Point metrics: precision={precision}, recall={recall}, f1={f1}")
    return PointMetrics(precision, recall, f1)
