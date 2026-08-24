from collections.abc import Sequence
from dataclasses import dataclass

from mark_ad_benchmark.metrics.affiliation import affiliation_metrics


@dataclass(frozen=True)
class EventMetrics:
    point_adjusted_f1: float
    event_f1: float
    range_f1: float
    affiliation_precision: float
    affiliation_recall: float
    affiliation_f1: float


def event_metrics(
    labels: Sequence[int], predictions: Sequence[int], recall_existence_weight: float
) -> EventMetrics:
    """Calculate event-aware metrics from one fixed binary prediction vector."""
    validate_binary_vectors(labels, predictions)
    adjusted = point_adjust_predictions(labels, predictions)
    point_adjusted_f1 = f1(labels, adjusted)
    precision = point_precision(labels, predictions)
    event_recall = event_hit_recall(labels, predictions)
    range_precision = range_score(predictions, labels, 0.0)
    range_recall = range_score(labels, predictions, recall_existence_weight)
    affiliation = affiliation_metrics(labels, predictions)
    return EventMetrics(
        point_adjusted_f1=point_adjusted_f1,
        event_f1=harmonic_mean(precision, event_recall),
        range_f1=harmonic_mean(range_precision, range_recall),
        affiliation_precision=affiliation.precision,
        affiliation_recall=affiliation.recall,
        affiliation_f1=harmonic_mean(affiliation.precision, affiliation.recall),
    )


def ranges(values: Sequence[int]) -> tuple[tuple[int, int], ...]:
    """Return contiguous positive runs as inclusive index ranges."""
    result = []
    start: int | None = None
    for index, value in enumerate(values):
        if value and start is None:
            start = index
        elif not value and start is not None:
            result.append((start, index - 1))
            start = None
    if start is not None:
        result.append((start, len(values) - 1))
    return tuple(result)


def point_adjust_predictions(
    labels: Sequence[int], predictions: Sequence[int]
) -> tuple[int, ...]:
    """Expand a hit prediction to cover its complete true anomaly range."""
    adjusted = list(predictions)
    for start, end in ranges(labels):
        if any(predictions[start : end + 1]):
            adjusted[start : end + 1] = [1] * (end - start + 1)
    return tuple(adjusted)


def event_hit_recall(labels: Sequence[int], predictions: Sequence[int]) -> float:
    """Return the fraction of true anomaly ranges hit by a prediction."""
    true_ranges = ranges(labels)
    return (
        sum(any(predictions[start : end + 1]) for start, end in true_ranges)
        / len(true_ranges)
        if true_ranges
        else 0.0
    )


def range_score(
    references: Sequence[int], predictions: Sequence[int], existence_weight: float
) -> float:
    """Calculate TSB-AD range recall semantics for binary reference ranges."""
    reference_ranges = ranges(references)
    predicted_ranges = ranges(predictions)
    if not reference_ranges:
        return 0.0
    total = 0.0
    for start, end in reference_ranges:
        overlaps = [
            predicted
            for predicted in predicted_ranges
            if predicted[0] <= end and start <= predicted[1]
        ]
        existence = float(bool(overlaps))
        covered = sum(predictions[start : end + 1]) / (end - start + 1)
        cardinality = 1 / len(overlaps) if overlaps else 0.0
        total += (
            existence_weight * existence
            + (1 - existence_weight) * covered * cardinality
        )
    return total / len(reference_ranges)


def point_precision(labels: Sequence[int], predictions: Sequence[int]) -> float:
    """Calculate point precision with a zero-prediction value of zero."""
    predicted = sum(predictions)
    return (
        sum(label and prediction for label, prediction in zip(labels, predictions))
        / predicted
        if predicted
        else 0.0
    )


def f1(labels: Sequence[int], predictions: Sequence[int]) -> float:
    """Calculate point F1 from aligned binary vectors."""
    precision = point_precision(labels, predictions)
    positives = sum(labels)
    recall = (
        sum(label and prediction for label, prediction in zip(labels, predictions))
        / positives
        if positives
        else 0.0
    )
    return harmonic_mean(precision, recall)


def harmonic_mean(first: float, second: float) -> float:
    """Return the harmonic mean, using zero when either component is absent."""
    return 2 * first * second / (first + second) if first + second else 0.0


def validate_binary_vectors(labels: Sequence[int], predictions: Sequence[int]) -> None:
    """Reject misaligned or nonbinary event-metric inputs."""
    if len(labels) != len(predictions):
        raise ValueError("Labels and predictions must have equal lengths")
    if any(value not in {0, 1} for value in (*labels, *predictions)):
        raise ValueError("Event metrics require binary vectors")
