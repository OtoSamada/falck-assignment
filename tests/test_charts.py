"""Specify plotted data and PNG output without image snapshots or network calls.

Plotters consume approved, sorted metrics and return caller-owned Figures.
"""

import re
from datetime import date
from itertools import combinations
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.image as mimage
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.text import Text
from pandas.testing import assert_frame_equal

from pln_fx import charts

CURRENCIES = ["EUR", "GBP", "JPY", "USD"]
DATES = [date(2025, 1, 10), date(2025, 1, 13), date(2025, 1, 14)]
RATES = {
    "EUR": [4.0, 6.0, 3.0],
    "GBP": [4.0, 4.0, 5.0],
    "JPY": [0.024123, 0.048246, 0.0361845],
    "USD": [2.0, 1.0, 3.0],
}
INDICES = {
    # Preserve the supplied float, even when recalculation differs by one ULP.
    "EUR": [100.0, 150.00000000000003, 75.0],
    "GBP": [100.0, 100.0, 125.0],
    "JPY": [100.0, 200.0, 150.0],
    "USD": [100.0, 50.0, 150.0],
}


@pytest.fixture
def metrics() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(DATES * 4),
            "table_no": ["006/A/NBP/2025", "007/A/NBP/2025", "008/A/NBP/2025"] * 4,
            "currency": [currency for currency in CURRENCIES for _ in DATES],
            "rate": [value for currency in CURRENCIES for value in RATES[currency]],
            "previous_date": pd.to_datetime([None, DATES[0], DATES[1]] * 4),
            "previous_rate": [
                None,
                4.0,
                6.0,
                None,
                4.0,
                4.0,
                None,
                0.024123,
                0.048246,
                None,
                2.0,
                1.0,
            ],
            "absolute_change": [
                None,
                2.0,
                -3.0,
                None,
                0.0,
                1.0,
                None,
                0.024123,
                -0.0120615,
                None,
                -1.0,
                2.0,
            ],
            "percentage_change": [
                None,
                50.0,
                -50.0,
                None,
                0.0,
                25.0,
                None,
                100.0,
                -25.0,
                None,
                -50.0,
                200.0,
            ],
            "indexed_rate": [
                value for currency in CURRENCIES for value in INDICES[currency]
            ],
        }
    ).astype({"currency": "string", "table_no": "string"})
    frame.index = pd.RangeIndex(20, 32, name="source_row")
    return frame


def test_historical_chart_preserves_rates_dates_units_and_independent_scales(
    metrics: pd.DataFrame,
) -> None:
    before = metrics.copy(deep=True)

    figure = charts.plot_historical_rates(metrics)

    try:
        assert isinstance(figure, Figure)
        axes = figure.get_axes()
        assert len(axes) == 4
        for axis, currency in zip(axes, CURRENCIES, strict=True):
            assert currency in axis.get_title()
            assert re.search(
                rf"\bPLN\s*(?:per|/)\s*1\s*{currency}\b", axis.get_ylabel()
            )
            assert len(axis.get_lines()) == 1
            line = axis.get_lines()[0]
            assert line.get_ydata().tolist() == RATES[currency]
            assert [
                value.date() for value in mdates.num2date(line.get_xdata(orig=False))
            ] == DATES
        for first, second in combinations(axes, 2):
            assert not first.get_shared_y_axes().joined(first, second)

        text = " ".join(item.get_text() for item in figure.findobj(Text)).lower()
        assert "scales" in text and ("independent" in text or "separate" in text)
        assert "2025-01-10" in text and "2025-01-14" in text
        assert any("date" in axis.get_xlabel().lower() for axis in axes)
        assert_frame_equal(metrics, before, check_exact=True)
    finally:
        figure.clear()
        plt.close(figure)


def test_indexed_chart_uses_supplied_indices_and_explains_appreciation(
    metrics: pd.DataFrame,
) -> None:
    before = metrics.copy(deep=True)

    figure = charts.plot_indexed_rates(metrics)

    try:
        assert isinstance(figure, Figure)
        assert len(figure.get_axes()) == 1
        axis = figure.get_axes()[0]
        series = [line for line in axis.get_lines() if line.get_label() in CURRENCIES]
        assert len(series) == 4
        assert {line.get_label() for line in series} == set(CURRENCIES)
        for line in series:
            assert line.get_ydata().tolist() == INDICES[line.get_label()]
            assert [
                value.date() for value in mdates.num2date(line.get_xdata(orig=False))
            ] == DATES

        references = [line for line in axis.get_lines() if line not in series]
        assert any(
            len(line.get_ydata()) >= 2
            and all(value == 100 for value in line.get_ydata())
            for line in references
        )
        legends = [*figure.legends]
        if axis.get_legend() is not None:
            legends.append(axis.get_legend())
        assert [
            item.get_text() for legend in legends for item in legend.get_texts()
        ] == CURRENCIES

        ylabel = axis.get_ylabel().lower()
        assert "index" in ylabel and "100" in ylabel
        assert "date" in axis.get_xlabel().lower()
        assert axis.get_title().strip() or any(
            item.get_text().strip() for item in figure.texts
        )
        text = " ".join(item.get_text() for item in figure.findobj(Text)).lower()
        assert "first" in text
        assert "appreciat" in text and "pln" in text
        assert any(word in text for word in ["higher", "rising", "increas"])
        assert "2025-01-10" in text and "2025-01-14" in text
        assert_frame_equal(metrics, before, check_exact=True)
    finally:
        figure.clear()
        plt.close(figure)


@pytest.mark.parametrize(
    ("function_name", "filename"),
    [
        pytest.param("plot_historical_rates", "historical_rates.png", id="historical"),
        pytest.param("plot_indexed_rates", "indexed_rates.png", id="indexed"),
    ],
)
def test_chart_saves_valid_png_with_agg_without_registering_pyplot_figures(
    tmp_path: Path,
    metrics: pd.DataFrame,
    function_name: str,
    filename: str,
) -> None:
    registered_before = set(plt.get_fignums())

    figure = getattr(charts, function_name)(metrics)

    try:
        assert isinstance(figure, Figure)
        assert isinstance(figure.canvas, FigureCanvasAgg)
        assert set(plt.get_fignums()) == registered_before
        path = tmp_path / filename
        figure.savefig(path, format="png", dpi=150, facecolor="white")

        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        image = mimage.imread(path)
        assert image.ndim == 3
        assert image.shape[0] > 0 and image.shape[1] > 0
    finally:
        figure.clear()
        plt.close(figure)

    assert set(plt.get_fignums()) == registered_before
