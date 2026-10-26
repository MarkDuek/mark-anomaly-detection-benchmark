from collections.abc import Sequence
from dataclasses import dataclass

from mark_ad_benchmark.logging import ComponentLogger
from mark_ad_benchmark.metrics.intervals import Interval, extract_intervals


@dataclass(frozen=True)
class RangeMetrics:
    precision: float
    recall: float
    f1: float


def range_metrics(
    labels: Sequence[int],
    predictions: Sequence[int],
    recall_existence_weight: float,
    logger: ComponentLogger,
) -> RangeMetrics:
    """Calculate range F1 from one fixed binary prediction vector."""

    # 1. extract label and predicted intervals (documentation excerpt; keep synchronized)
    label_intervals = extract_intervals(labels)
    predicted_intervals = extract_intervals(predictions)

    recall = range_recall(label_intervals, predicted_intervals, recall_existence_weight)
    precision = range_precision(label_intervals, predicted_intervals)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    logger.info(f"Evaluated range F1: {f1}")
    return RangeMetrics(precision=precision, recall=recall, f1=f1)


def _range_reward(
    interval: Interval,
    comparison_intervals: Sequence[Interval],
    existence_weight: float,
) -> float:
    # 2. score existence, coverage, and cardinality (documentation excerpt; keep synchronized)
    start, end = interval
    overlaps = [
        compared
        for compared in comparison_intervals
        if compared[0] <= end and start <= compared[1]
    ]
    existence = float(bool(overlaps))
    covered = sum(
        min(end, compared[1]) - max(start, compared[0]) + 1 for compared in overlaps
    ) / (end - start + 1)
    cardinality = 1 / len(overlaps) if overlaps else 0.0
    return existence_weight * existence + (1 - existence_weight) * covered * cardinality


def range_recall(
    label_intervals: Sequence[Interval],
    predicted_intervals: Sequence[Interval],
    existence_weight: float,
) -> float:
    """Average existence and overlap rewards over label intervals."""
    # 3. average rewards over label intervals (documentation excerpt; keep synchronized)
    return (
        sum(
            _range_reward(label, predicted_intervals, existence_weight)
            for label in label_intervals
        )
        / len(label_intervals)
        if label_intervals
        else 0.0
    )


def range_precision(
    label_intervals: Sequence[Interval], predicted_intervals: Sequence[Interval]
) -> float:
    """Average overlap rewards over predicted intervals without existence credit."""
    # 4. average rewards over predicted intervals (documentation excerpt; keep synchronized)
    return (
        sum(
            _range_reward(predicted, label_intervals, 0.0)
            for predicted in predicted_intervals
        )
        / len(predicted_intervals)
        if predicted_intervals
        else 0.0
    )
