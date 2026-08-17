import math
from collections.abc import Sequence

from mark_ad_benchmark.logging import ComponentLogger
from mark_ad_benchmark.thresholds.base import ThresholdSelector


class BestF1ThresholdSelector(ThresholdSelector):
    """Choose the highest validation-score threshold with maximal point F1."""

    def __init__(self, logger: ComponentLogger) -> None:
        self.logger = logger

    def fit(self, scores: Sequence[float], labels: Sequence[int]) -> float:
        """Select a threshold using labeled validation scores."""
        validate_inputs(scores, labels, self.logger)
        threshold = max(
            {*scores, math.inf},
            key=lambda candidate: (point_f1(scores, labels, candidate), candidate),
        )
        self.logger.info(f"Selected validation F1 threshold: {threshold}")
        return threshold


def validate_inputs(
    scores: Sequence[float], labels: Sequence[int], logger: ComponentLogger
) -> None:
    if len(scores) != len(labels):
        message = "Threshold selection requires scores and labels of equal length"
        logger.error(message)
        raise ValueError(message)
    if any(label not in {0, 1} for label in labels):
        message = "Threshold selection requires binary labels"
        logger.error(message)
        raise ValueError(message)
    if any(not math.isfinite(score) for score in scores):
        message = "Threshold selection requires finite scores"
        logger.error(message)
        raise ValueError(message)


def point_f1(scores: Sequence[float], labels: Sequence[int], threshold: float) -> float:
    true_positives = sum(
        label == 1 and score >= threshold for score, label in zip(scores, labels)
    )
    false_positives = sum(
        label == 0 and score >= threshold for score, label in zip(scores, labels)
    )
    false_negatives = sum(
        label == 1 and score < threshold for score, label in zip(scores, labels)
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
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0
