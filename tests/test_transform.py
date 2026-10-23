"""Dataset checks on validated NBP models; schema validation belongs to nbp.py."""

from datetime import date

import pandas as pd
import pytest
from pandas.api.types import is_datetime64_dtype
from pandas.testing import assert_frame_equal

from pln_fx import transform
from pln_fx.nbp import NBPRate, NBPTable


@pytest.fixture
def tables() -> list[NBPTable]:
    """Unsorted synthetic publications with a weekend gap and an extra currency."""
    return [
        NBPTable(
            table="A",
            no="007/A/NBP/2025",
            effectiveDate=date(2025, 1, 13),
            rates=[
                NBPRate(code="USD", mid=3.887654),
                NBPRate(code="JPY", mid=0.024321),
                NBPRate(code="CHF", mid=4.6),
                NBPRate(code="EUR", mid=4.223456),
                NBPRate(code="GBP", mid=5.223456),
            ],
        ),
        NBPTable(
            table="A",
            no="006/A/NBP/2025",
            effectiveDate=date(2025, 1, 10),
            rates=[
                NBPRate(code="JPY", mid=0.024123),
                NBPRate(code="GBP", mid=5.123456),
                NBPRate(code="EUR", mid=4.123456),
                NBPRate(code="CHF", mid=4.5),
                NBPRate(code="USD", mid=3.987654),
            ],
        ),
    ]


def test_selects_targets_preserves_rates_and_gaps_and_sorts(
    tables: list[NBPTable],
) -> None:
    frame = transform.to_rates_frame(tables)

    expected = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-10",
                    "2025-01-13",
                    "2025-01-10",
                    "2025-01-13",
                    "2025-01-10",
                    "2025-01-13",
                    "2025-01-10",
                    "2025-01-13",
                ]
            ),
            "table_no": ["006/A/NBP/2025", "007/A/NBP/2025"] * 4,
            "currency": ["EUR", "EUR", "GBP", "GBP", "JPY", "JPY", "USD", "USD"],
            "rate": [
                4.123456,
                4.223456,
                5.123456,
                5.223456,
                0.024123,
                0.024321,
                3.987654,
                3.887654,
            ],
        }
    )

    # Check values exactly; check logical dtypes without fixing storage or time units.
    assert_frame_equal(frame, expected, check_dtype=False, check_exact=True)
    assert is_datetime64_dtype(frame["date"].dtype)
    assert isinstance(frame["table_no"].dtype, pd.StringDtype)
    assert isinstance(frame["currency"].dtype, pd.StringDtype)
    assert frame["rate"].dtype == "float64"
    assert isinstance(frame.index, pd.RangeIndex)


@pytest.mark.parametrize("missing_code", ["EUR", "USD", "GBP", "JPY"])
def test_requires_each_target_in_every_publication(
    tables: list[NBPTable], missing_code: str
) -> None:
    tables[0].rates = [rate for rate in tables[0].rates if rate.code != missing_code]

    with pytest.raises(ValueError):
        transform.to_rates_frame(tables)


@pytest.mark.parametrize(
    "duplicate_mid",
    [
        pytest.param(4.123456, id="identical-rate"),
        pytest.param(4.999999, id="conflicting-rate"),
    ],
)
def test_rejects_duplicate_target_within_a_publication(
    tables: list[NBPTable], duplicate_mid: float
) -> None:
    tables[1].rates.append(NBPRate(code="EUR", mid=duplicate_mid))

    with pytest.raises(ValueError):
        transform.to_rates_frame(tables)


@pytest.mark.parametrize(
    "duplicate_mid",
    [
        pytest.param(4.123456, id="identical-rate"),
        pytest.param(4.999999, id="conflicting-rate"),
    ],
)
def test_rejects_duplicate_currency_dates_across_different_table_numbers(
    tables: list[NBPTable], duplicate_mid: float
) -> None:
    duplicate = tables[1].model_copy(deep=True)
    duplicate.no = "999/A/NBP/2025"
    next(rate for rate in duplicate.rates if rate.code == "EUR").mid = duplicate_mid

    with pytest.raises(ValueError):
        transform.to_rates_frame([tables[1], duplicate])


def test_rejects_non_target_duplicates_before_filtering(tables: list[NBPTable]) -> None:
    tables[1].rates.append(NBPRate(code="CHF", mid=4.5))

    with pytest.raises(ValueError):
        transform.to_rates_frame(tables)


def test_accepts_one_publication_with_one_observation_per_target(
    tables: list[NBPTable],
) -> None:
    publication = tables[1]
    publication.rates = [rate for rate in publication.rates if rate.code != "CHF"]

    frame = transform.to_rates_frame([publication])

    assert len(frame) == 4
    assert frame["currency"].tolist() == ["EUR", "GBP", "JPY", "USD"]
    assert frame["date"].tolist() == [pd.Timestamp("2025-01-10")] * 4


def test_rejects_an_empty_dataset() -> None:
    with pytest.raises(ValueError):
        transform.to_rates_frame([])
