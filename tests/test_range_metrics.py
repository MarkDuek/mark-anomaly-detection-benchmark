import pytest

from mark_ad_benchmark.logging import Logger
from mark_ad_benchmark.metrics.range import range_metrics


def test_range_f1_preserves_existing_overlap_score() -> None:
    metrics = range_metrics(
        (0, 1, 1, 0),
        (0, 1, 0, 0),
        0.2,
        Logger().for_component("mark_ad_benchmark.metrics.range"),
    )

    assert metrics.f1 == pytest.approx(0.75)
