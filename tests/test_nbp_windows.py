"""Specify inclusive, nonoverlapping NBP request windows."""

from datetime import date, timedelta
from itertools import pairwise

import pytest

from pln_fx import nbp


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        pytest.param(
            date(2024, 1, 1),
            date(2024, 1, 1),
            [(date(2024, 1, 1), date(2024, 1, 1))],
            id="single-day",
        ),
        pytest.param(
            date(2024, 1, 1),
            date(2024, 3, 30),
            [(date(2024, 1, 1), date(2024, 3, 30))],
            id="exactly-90-days",
        ),
        pytest.param(
            date(2024, 1, 1),
            date(2024, 3, 31),
            [
                (date(2024, 1, 1), date(2024, 3, 30)),
                (date(2024, 3, 31), date(2024, 3, 31)),
            ],
            id="91-days",
        ),
        pytest.param(
            date(2024, 1, 1),
            date(2024, 6, 28),
            [
                (date(2024, 1, 1), date(2024, 3, 30)),
                (date(2024, 3, 31), date(2024, 6, 28)),
            ],
            id="exactly-180-days",
        ),
        pytest.param(
            date(2024, 1, 1),
            date(2024, 6, 29),
            [
                (date(2024, 1, 1), date(2024, 3, 30)),
                (date(2024, 3, 31), date(2024, 6, 28)),
                (date(2024, 6, 29), date(2024, 6, 29)),
            ],
            id="181-days",
        ),
        pytest.param(
            date(2024, 1, 1),
            date(2024, 4, 5),
            [
                (date(2024, 1, 1), date(2024, 3, 30)),
                (date(2024, 3, 31), date(2024, 4, 5)),
            ],
            id="short-final-window",
        ),
        pytest.param(
            date(2024, 2, 28),
            date(2024, 3, 1),
            [(date(2024, 2, 28), date(2024, 3, 1))],
            id="leap-day-and-month-boundary",
        ),
        pytest.param(
            date(2023, 12, 30),
            date(2024, 1, 2),
            [(date(2023, 12, 30), date(2024, 1, 2))],
            id="year-boundary",
        ),
        pytest.param(
            date.max,
            date.max,
            [(date.max, date.max)],
            id="last-representable-day",
        ),
        pytest.param(
            date(9999, 9, 30),
            date.max,
            [
                (date(9999, 9, 30), date(9999, 12, 28)),
                (date(9999, 12, 29), date.max),
            ],
            id="multiple-windows-ending-at-date-max",
        ),
    ],
)
def test_date_windows_match_inclusive_boundaries(
    start: date, end: date, expected: list[tuple[date, date]]
) -> None:
    assert list(nbp.iter_date_windows(start, end)) == expected


def test_date_windows_cover_five_years_without_gaps_or_overlaps() -> None:
    start = date(2021, 10, 23)
    end = date(2026, 10, 22)

    windows = list(nbp.iter_date_windows(start, end))

    assert windows[0][0] == start
    assert windows[-1][1] == end
    assert all(1 <= (last - first).days + 1 <= 90 for first, last in windows)
    assert all(
        following_start == previous_end + timedelta(days=1)
        for (_, previous_end), (following_start, _) in pairwise(windows)
    )
    assert (
        sum((last - first).days + 1 for first, last in windows)
        == (end - start).days + 1
    )


def test_date_windows_reject_start_after_end() -> None:
    with pytest.raises(ValueError):
        list(nbp.iter_date_windows(date(2024, 3, 31), date(2024, 3, 30)))
