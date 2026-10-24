"""Specify metrics between consecutive published NBP average-rate observations.

Inputs use the ingestion schema. Numerical expectations below are literal,
independently calculated results; no ingestion validators are repeated.
"""

import pandas as pd
import pytest
from pandas.api.types import is_datetime64_dtype
from pandas.testing import assert_frame_equal, assert_series_equal

from pln_fx import analysis

INPUT_COLUMNS = ["date", "table_no", "currency", "rate"]
METRIC_COLUMNS = [
    "previous_rate",
    "absolute_change",
    "percentage_change",
    "indexed_rate",
]
OUTPUT_COLUMNS = [*INPUT_COLUMNS, "previous_date", *METRIC_COLUMNS]
SUMMARY_COLUMNS = [
    "currency",
    "finding",
    "date",
    "table_no",
    "rate",
    "previous_date",
    "previous_rate",
    "absolute_change",
    "percentage_change",
    "indexed_rate",
]
PUBLICATION_DATES = pd.to_datetime(
    [
        "2025-01-10",
        "2025-01-13",
        "2025-01-14",
        "2025-01-15",
        "2025-01-16",
        "2025-01-17",
        "2025-01-20",
    ]
)
TABLE_NUMBERS = [
    "006/A/NBP/2025",
    "007/A/NBP/2025",
    "008/A/NBP/2025",
    "009/A/NBP/2025",
    "010/A/NBP/2025",
    "011/A/NBP/2025",
    "012/A/NBP/2025",
]
RATES_BY_CURRENCY = {
    "EUR": [4.0, 6.0, 5.0, 2.0, 4.0, 8.0, 4.0],
    "USD": [2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
    "GBP": [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
    "JPY": [0.024123, 0.048246, 0.024123, 0.0120615, 0.024123, 0.048246, 0.024123],
}


@pytest.fixture
def rates() -> pd.DataFrame:
    rows = [
        (day, table_no, currency, values[position])
        for position, (day, table_no) in enumerate(
            zip(PUBLICATION_DATES, TABLE_NUMBERS, strict=True)
        )
        for currency, values in RATES_BY_CURRENCY.items()
    ]
    frame = pd.DataFrame(rows, columns=INPUT_COLUMNS).astype(
        {"table_no": "string", "currency": "string", "rate": "float64"}
    )
    frame.index = pd.RangeIndex(100, 100 + len(frame), name="source_row")
    return frame.iloc[::-1]


def test_metrics_match_literal_results_and_preserve_input(rates: pd.DataFrame) -> None:
    before = rates.copy(deep=True)

    metrics = analysis.calculate_rate_metrics(rates)

    assert_frame_equal(rates, before, check_exact=True)
    assert metrics.columns.tolist() == OUTPUT_COLUMNS
    assert isinstance(metrics.index, pd.RangeIndex)
    for column in ["date", "previous_date"]:
        assert is_datetime64_dtype(metrics[column].dtype)
    for column in ["table_no", "currency"]:
        assert isinstance(metrics[column].dtype, pd.StringDtype)
    for column in ["rate", *METRIC_COLUMNS]:
        assert metrics[column].dtype == "float64"

    expected_observations = pd.DataFrame(
        {
            "date": PUBLICATION_DATES.tolist() * 4,
            "table_no": TABLE_NUMBERS * 4,
            "currency": ["EUR"] * 7 + ["GBP"] * 7 + ["JPY"] * 7 + ["USD"] * 7,
            "rate": [
                *RATES_BY_CURRENCY["EUR"],
                *RATES_BY_CURRENCY["GBP"],
                *RATES_BY_CURRENCY["JPY"],
                *RATES_BY_CURRENCY["USD"],
            ],
        }
    )
    assert_frame_equal(
        metrics[INPUT_COLUMNS],
        expected_observations,
        check_dtype=False,
        check_exact=True,
    )
    previous_dates = pd.Series(
        pd.to_datetime(
            [
                None,
                "2025-01-10",
                "2025-01-13",
                "2025-01-14",
                "2025-01-15",
                "2025-01-16",
                "2025-01-17",
            ]
        ),
        name="previous_date",
    )
    # Each tuple is (previous rate, absolute change, percentage change, index).
    expected = {
        "EUR": [
            (None, None, None, 100),
            (4, 2, 50, 150),
            (6, -1, -50 / 3, 125),
            (5, -3, -60, 50),
            (2, 2, 100, 100),
            (4, 4, 100, 200),
            (8, -4, -50, 100),
        ],
        "GBP": [
            (None, None, None, 100),
            (4, 0, 0, 100),
            (4, 0, 0, 100),
            (4, 0, 0, 100),
            (4, 0, 0, 100),
            (4, 0, 0, 100),
            (4, 0, 0, 100),
        ],
        "JPY": [
            (None, None, None, 100),
            (0.024123, 0.024123, 100, 200),
            (0.048246, -0.024123, -50, 100),
            (0.024123, -0.0120615, -50, 50),
            (0.0120615, 0.0120615, 100, 100),
            (0.024123, 0.024123, 100, 200),
            (0.048246, -0.024123, -50, 100),
        ],
        "USD": [
            (None, None, None, 100),
            (2, -1, -50, 50),
            (1, 1, 100, 100),
            (2, -1, -50, 50),
            (1, 1, 100, 100),
            (2, -1, -50, 50),
            (1, 1, 100, 100),
        ],
    }
    for currency, rows in expected.items():
        actual = metrics.loc[metrics["currency"] == currency].reset_index(drop=True)
        expected_metrics = pd.DataFrame(rows, columns=METRIC_COLUMNS)
        assert_series_equal(
            actual["previous_date"], previous_dates, check_dtype=False, check_exact=True
        )
        assert_series_equal(
            actual["previous_rate"],
            expected_metrics["previous_rate"],
            check_dtype=False,
            check_exact=True,
        )
        assert_frame_equal(
            actual[METRIC_COLUMNS[1:]],
            expected_metrics[METRIC_COLUMNS[1:]],
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )


def test_preserves_a_publication_gap_spanning_a_weekend_and_holiday(
    rates: pd.DataFrame,
) -> None:
    short = rates.loc[rates["date"].isin(PUBLICATION_DATES[:2])].copy()
    short["date"] = short["date"].map(
        {
            pd.Timestamp("2025-01-10"): pd.Timestamp("2025-01-03"),
            pd.Timestamp("2025-01-13"): pd.Timestamp("2025-01-07"),
        }
    )

    metrics = analysis.calculate_rate_metrics(short)

    assert (
        metrics["date"].tolist()
        == [pd.Timestamp("2025-01-03"), pd.Timestamp("2025-01-07")] * 4
    )
    assert_series_equal(
        metrics["previous_date"],
        pd.Series(pd.to_datetime([None, "2025-01-03"] * 4), name="previous_date"),
        check_dtype=False,
        check_exact=True,
    )
    eur = metrics.loc[metrics["currency"] == "EUR"]
    assert eur["absolute_change"].iloc[1] == pytest.approx(2.0)
    assert eur["percentage_change"].iloc[1] == pytest.approx(50.0)


def test_rejects_empty_input(rates: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        analysis.calculate_rate_metrics(rates.iloc[:0])


@pytest.mark.parametrize("case", ["single-publication", "one-short-currency"])
def test_requires_at_least_two_observations_per_currency(
    rates: pd.DataFrame,
    case: str,
) -> None:
    first_publication = rates["date"] == PUBLICATION_DATES[0]
    short = rates.loc[
        first_publication
        if case == "single-publication"
        else (rates["currency"] != "EUR") | first_publication
    ]

    with pytest.raises(ValueError):
        analysis.calculate_rate_metrics(short)


@pytest.mark.parametrize(
    "values",
    [
        pytest.param([1e300, 1e-300, 1e300], id="percentage-overflow"),
        pytest.param(
            [1.0, 1e200, 1e307], id="index-overflow-with-finite-percentage-changes"
        ),
    ],
)
def test_rejects_nonfinite_calculations_from_finite_positive_rates(
    rates: pd.DataFrame,
    values: list[float],
) -> None:
    short = rates.loc[rates["date"].isin(PUBLICATION_DATES[:3])].copy()
    short["rate"] = short["date"].map(
        dict(zip(PUBLICATION_DATES[:3], values, strict=True))
    )

    with pytest.raises(ValueError):
        analysis.calculate_rate_metrics(short)


def test_summary_selects_literal_findings_and_preserves_context(
    rates: pd.DataFrame,
) -> None:
    metrics = analysis.calculate_rate_metrics(rates)
    before = metrics.copy(deep=True)

    summary = analysis.summarize_rates(metrics)

    assert_frame_equal(metrics, before, check_exact=True)
    assert summary.columns.tolist() == SUMMARY_COLUMNS
    for column in ["currency", "finding", "table_no"]:
        assert isinstance(summary[column].dtype, pd.StringDtype)
    for column in ["date", "previous_date"]:
        assert is_datetime64_dtype(summary[column].dtype)
    for column in ["rate", *METRIC_COLUMNS]:
        assert summary[column].dtype == "float64"

    expected = pd.DataFrame(
        [
            ("EUR", "min_rate", "2025-01-15", 2.0),
            ("EUR", "max_rate", "2025-01-17", 8.0),
            ("EUR", "largest_fall", "2025-01-15", 2.0),
            ("EUR", "largest_rise", "2025-01-16", 4.0),
            ("GBP", "min_rate", "2025-01-10", 4.0),
            ("GBP", "max_rate", "2025-01-10", 4.0),
            ("GBP", "largest_fall", None, None),
            ("GBP", "largest_rise", None, None),
            ("JPY", "min_rate", "2025-01-15", 0.0120615),
            ("JPY", "max_rate", "2025-01-13", 0.048246),
            ("JPY", "largest_fall", "2025-01-14", 0.024123),
            ("JPY", "largest_rise", "2025-01-13", 0.048246),
            ("USD", "min_rate", "2025-01-13", 1.0),
            ("USD", "max_rate", "2025-01-10", 2.0),
            ("USD", "largest_fall", "2025-01-13", 1.0),
            ("USD", "largest_rise", "2025-01-14", 2.0),
        ],
        columns=["currency", "finding", "date", "rate"],
    )
    expected["date"] = pd.to_datetime(expected["date"])
    assert_frame_equal(
        summary[["currency", "finding", "date", "rate"]],
        expected,
        check_dtype=False,
        check_exact=True,
    )
    for _, row in summary.loc[summary["date"].notna()].iterrows():
        source = metrics.loc[
            (metrics["currency"] == row["currency"]) & (metrics["date"] == row["date"])
        ].iloc[0]
        assert_series_equal(
            row[OUTPUT_COLUMNS],
            source,
            check_names=False,
            check_dtype=False,
            check_exact=True,
        )
    absent = summary.loc[summary["date"].isna()]
    assert absent.drop(columns=["currency", "finding"]).isna().all().all()


@pytest.mark.parametrize(
    ("values", "missing_finding", "present_finding", "event_date", "percentage"),
    [
        pytest.param(
            [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0],
            "largest_fall",
            "largest_rise",
            "2025-01-14",
            100.0,
            id="rises-and-plateaus",
        ),
        pytest.param(
            [4.0, 4.0, 3.0, 3.0, 2.0, 2.0, 1.0],
            "largest_rise",
            "largest_fall",
            "2025-01-20",
            -50.0,
            id="falls-and-plateaus",
        ),
    ],
)
def test_summary_leaves_an_unobserved_direction_null(
    rates: pd.DataFrame,
    values: list[float],
    missing_finding: str,
    present_finding: str,
    event_date: str,
    percentage: float,
) -> None:
    source = rates.copy(deep=True)
    gbp = source["currency"] == "GBP"
    source.loc[gbp, "rate"] = source.loc[gbp, "date"].map(
        dict(zip(PUBLICATION_DATES, values, strict=True))
    )
    metrics = analysis.calculate_rate_metrics(source)

    summary = analysis.summarize_rates(metrics)

    findings = summary.loc[summary["currency"] == "GBP"].set_index("finding")
    assert findings.loc[missing_finding].drop(labels="currency").isna().all()
    present = findings.loc[present_finding]
    assert present["date"] == pd.Timestamp(event_date)
    assert present["percentage_change"] == percentage


def test_summary_does_not_merge_near_ties_by_rounding_or_tolerance(
    rates: pd.DataFrame,
) -> None:
    source = rates.copy(deep=True)
    usd = source["currency"] == "USD"
    source.loc[usd & (source["date"] == pd.Timestamp("2025-01-17")), "rate"] = (
        0.999999999999
    )
    source.loc[usd & (source["date"] == pd.Timestamp("2025-01-20")), "rate"] = (
        2.000000000001
    )
    metrics = analysis.calculate_rate_metrics(source)

    summary = analysis.summarize_rates(metrics)

    expected = pd.DataFrame(
        [
            ("min_rate", "2025-01-17", 0.999999999999),
            ("max_rate", "2025-01-20", 2.000000000001),
            ("largest_fall", "2025-01-17", 0.999999999999),
            ("largest_rise", "2025-01-20", 2.000000000001),
        ],
        columns=["finding", "date", "rate"],
    )
    expected["date"] = pd.to_datetime(expected["date"])
    actual = summary.loc[
        summary["currency"] == "USD", ["finding", "date", "rate"]
    ].reset_index(drop=True)
    assert_frame_equal(actual, expected, check_dtype=False, check_exact=True)


def test_summary_preserves_supplied_metrics_without_recalculating(
    rates: pd.DataFrame,
) -> None:
    metrics = analysis.calculate_rate_metrics(rates)
    selected = (metrics["currency"] == "EUR") & (
        metrics["date"] == pd.Timestamp("2025-01-17")
    )
    columns = ["absolute_change", "percentage_change", "indexed_rate"]
    # Equivalent floating-point evaluation orders can differ by one ULP.
    supplied = [4.000000000000001, 100.00000000000001, 200.00000000000003]
    metrics.loc[selected, columns] = supplied

    summary = analysis.summarize_rates(metrics)

    rise = summary.loc[
        (summary["currency"] == "EUR") & (summary["finding"] == "largest_rise")
    ]
    assert rise["date"].tolist() == [pd.Timestamp("2025-01-17")]
    assert rise.iloc[0][columns].tolist() == supplied
