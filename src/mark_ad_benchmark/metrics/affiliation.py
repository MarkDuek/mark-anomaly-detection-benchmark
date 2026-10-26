"""TSB-AD-style affiliation precision and recall for binary time series."""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

Interval = tuple[float, float]


@dataclass(frozen=True)
class AffiliationMetrics:
    precision: float
    recall: float
    f1: float


def affiliation_metrics(
    labels: Sequence[int], predictions: Sequence[int]
) -> AffiliationMetrics:
    """Calculate affiliation precision and recall from binary vectors."""
    # 1. extract affiliation events and time range (documentation excerpt; keep synchronized)
    true_events = _events(labels)
    predicted_events = _events(predictions)
    if not true_events or not predicted_events:
        return AffiliationMetrics(precision=0.0, recall=0.0, f1=0.0)

    time_range = (0.0, float(len(labels)))
    true_zones = _zones(true_events, time_range)
    precision_scores = []
    # 7. average precision rewards across populated label zones (documentation excerpt; keep synchronized)
    for event, zone in zip(true_events, true_zones):
        predicted_parts = [
            part
            for prediction in predicted_events
            if (part := _intersection(prediction, zone)) is not None
        ]
        predicted_length = sum(_length(part) for part in predicted_parts)
        if predicted_length:
            score = (
                sum(_precision_integral(part, event, zone) for part in predicted_parts)
                / predicted_length
            )
            precision_scores.append(score)

    recall_scores = []
    # 11. average recall rewards within each label zone (documentation excerpt; keep synchronized)
    for event, zone in zip(true_events, true_zones):
        predicted_parts = [
            part
            for prediction in predicted_events
            if (part := _intersection(prediction, zone)) is not None
        ]
        score = 0.0
        for prediction, prediction_zone in zip(
            predicted_parts, _zones(predicted_parts, zone)
        ):
            part = _intersection(event, prediction_zone)
            if part is not None:
                score += _recall_integral(prediction, part, zone)
        recall_scores.append(score / _length(event))

    precision = sum(precision_scores) / len(precision_scores)
    recall = sum(recall_scores) / len(recall_scores)
    # 12. calculate affiliation F1 (documentation excerpt; keep synchronized)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return AffiliationMetrics(precision, recall, f1)


def _events(values: Sequence[int]) -> list[Interval]:
    # 2. convert positive timestamps to half-open events (documentation excerpt; keep synchronized)
    events = []
    start: int | None = None
    for index, value in enumerate(values):
        if value and start is None:
            start = index
        elif not value and start is not None:
            events.append((float(start), float(index)))
            start = None
    if start is not None:
        events.append((float(start), float(len(values))))
    return events


def _zones(events: list[Interval], time_range: Interval) -> list[Interval]:
    # 3. build nearest-event affiliation zones (documentation excerpt; keep synchronized)
    zones = []
    for index, event in enumerate(events):
        left = time_range[0] if index == 0 else (events[index - 1][1] + event[0]) / 2
        right = (
            time_range[1]
            if index == len(events) - 1
            else (event[1] + events[index + 1][0]) / 2
        )
        zones.append((left, right))
    return zones


def _intersection(first: Interval, second: Interval) -> Interval | None:
    # 4. intersect half-open intervals (documentation excerpt; keep synchronized)
    start, end = max(first[0], second[0]), min(first[1], second[1])
    return (start, end) if start < end else None


def _length(interval: Interval) -> float:
    return interval[1] - interval[0]


def _distance_integral(start: float, end: float, limit: float) -> float:
    """Integrate ``d + min(d, limit)`` for distances from start through end."""
    # 5. integrate the capped precision-distance term (documentation excerpt; keep synchronized)
    linear = (end**2 - start**2) / 2
    capped = (min(end, limit) ** 2 - min(start, limit) ** 2) / 2
    capped += limit * (max(end - limit, 0.0) - max(start - limit, 0.0))
    return linear + capped


def _precision_integral(prediction: Interval, event: Interval, zone: Interval) -> float:
    # 6. integrate a predicted interval's precision reward (documentation excerpt; keep synchronized)
    overlap = _intersection(prediction, event)
    score = _length(overlap) if overlap is not None else 0.0
    zone_length = _length(zone)
    limit = min(event[0] - zone[0], zone[1] - event[1])
    left = _intersection(prediction, (zone[0], event[0]))
    right = _intersection(prediction, (event[1], zone[1]))
    for part, pivot in ((left, event[0]), (right, event[1])):
        if part is not None:
            distance_start = abs(part[0] - pivot)
            distance_end = abs(part[1] - pivot)
            score += (
                _length(part)
                - (
                    _distance_integral(
                        min(distance_start, distance_end),
                        max(distance_start, distance_end),
                        limit,
                    )
                    + _length(event) * _length(part)
                )
                / zone_length
            )
    return score


def _recall_integral(
    prediction: Interval, event_part: Interval, zone: Interval
) -> float:
    # 10. integrate a label interval's recall reward (documentation excerpt; keep synchronized)
    overlap = _intersection(prediction, event_part)
    score = _length(overlap) if overlap is not None else 0.0
    left = _intersection(event_part, (zone[0], prediction[0]))
    right = _intersection(event_part, (prediction[1], zone[1]))
    for part, pivot in ((left, prediction[0]), (right, prediction[1])):
        if part is not None and zone[0] < pivot < zone[1]:
            score += _recall_probability_integral(part, pivot, zone)
    return score


def _recall_probability_integral(
    interval: Interval, prediction_edge: float, zone: Interval
) -> float:
    # 9. integrate piecewise-linear recall probabilities (documentation excerpt; keep synchronized)
    breaks = sorted(
        {
            interval[0],
            interval[1],
            (zone[0] + zone[1]) / 2,
            (zone[0] + prediction_edge) / 2,
            (prediction_edge + zone[1]) / 2,
            prediction_edge,
        }
    )
    score = 0.0
    for start, end in pairwise(breaks):
        start, end = max(start, interval[0]), min(end, interval[1])
        if start >= end:
            continue
        score += (
            (end - start)
            * (
                _recall_probability(start, prediction_edge, zone)
                + _recall_probability(end, prediction_edge, zone)
            )
            / 2
        )
    return score


def _recall_probability(point: float, prediction_edge: float, zone: Interval) -> float:
    # 8. calculate the recall probability at one point (documentation excerpt; keep synchronized)
    distance = abs(point - prediction_edge)
    random_distance = min(point - zone[0], zone[1] - point)
    return 1 - (distance + min(distance, random_distance)) / _length(zone)
