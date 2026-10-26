from collections.abc import Sequence

Interval = tuple[int, int]


def extract_intervals(values: Sequence[int]) -> tuple[Interval, ...]:
    """Return maximal positive runs as inclusive, zero-based intervals."""

    # extract maximal contiguous positive intervals (documentation excerpt; keep synchronized)
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
