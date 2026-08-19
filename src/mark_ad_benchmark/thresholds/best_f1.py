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
        best_threshold = math.inf
        best_f1 = 0.0
        true_positives = 0
        false_positives = 0
        positives = sum(labels)
        ordered = sorted(zip(scores, labels), reverse=True)
        index = 0
        while index < len(ordered):
            threshold = ordered[index][0]
            while index < len(ordered) and ordered[index][0] == threshold:
                if ordered[index][1]:
                    true_positives += 1
                else:
                    false_positives += 1
                index += 1
            false_negatives = positives - true_positives
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
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        self.logger.info(f"Selected validation F1 threshold: {best_threshold}")
        return best_threshold


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
