"""Static charts for exchange rate levels and relative performance."""

import matplotlib.dates as mdates
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

_CURRENCY_STYLES = [
    ("EUR", "#0072B2", "-"),
    ("GBP", "#D55E00", "--"),
    ("JPY", "#009E73", "-."),
    ("USD", "#7B3294", ":"),
]


def _format_axis(axis: Axes) -> None:
    """Apply common date formatting and restrained chart styling."""
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axis.tick_params(labelsize=10)
    axis.margins(x=0.01)
    axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#DCE1E8", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)


def plot_historical_rates(metrics: pd.DataFrame) -> Figure:
    """Plot original PLN-per-one-unit rates from validated, sorted metrics.

    Return a caller-owned Agg figure; preserve inputs and publication gaps.
    """
    figure = Figure(figsize=(12, 8), dpi=150, layout="constrained", facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2, sharex=True)
    period = f"{metrics['date'].min():%Y-%m-%d} to {metrics['date'].max():%Y-%m-%d}"
    figure.suptitle(f"Historical exchange rates against PLN\n{period}", fontsize=15)
    figure.supxlabel(
        "Independent y-axis scales; lines connect published observations across gaps.",
        fontsize=9,
        color="#555555",
    )

    for axis, (currency, colour, style) in zip(
        axes.flat, _CURRENCY_STYLES, strict=True
    ):
        observations = metrics.loc[metrics["currency"] == currency]
        axis.plot(
            observations["date"],
            observations["rate"],
            color=colour,
            linestyle=style,
            linewidth=2,
        )
        axis.set_title(currency, fontsize=12, fontweight="bold")
        axis.set_ylabel(f"PLN per 1 {currency}", fontsize=11)
        _format_axis(axis)
    for axis in axes[-1]:
        axis.set_xlabel("Publication date", fontsize=11)
    return figure


def plot_indexed_rates(metrics: pd.DataFrame) -> Figure:
    """Plot supplied base-100 indices without recomputing analytical values.

    Return a caller-owned Agg figure; preserve inputs and publication gaps.
    """
    figure = Figure(figsize=(12, 6), dpi=150, layout="constrained", facecolor="white")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    period = f"{metrics['date'].min():%Y-%m-%d} to {metrics['date'].max():%Y-%m-%d}"
    figure.suptitle(f"Indexed currency performance against PLN\n{period}", fontsize=15)
    figure.supxlabel(
        "Higher index = foreign currency appreciation against PLN.\n"
        "Lines connect published observations across gaps.",
        fontsize=9,
        color="#555555",
    )

    for currency, colour, style in _CURRENCY_STYLES:
        observations = metrics.loc[metrics["currency"] == currency]
        axis.plot(
            observations["date"],
            observations["indexed_rate"],
            color=colour,
            linestyle=style,
            linewidth=2,
            label=currency,
        )
    axis.axhline(100, color="#7C8793", linestyle="--", linewidth=1, zorder=0)
    axis.set_ylabel("Index (first publication = 100)", fontsize=11)
    axis.set_xlabel("Publication date", fontsize=11)
    axis.legend(
        loc="lower left", bbox_to_anchor=(0, 1.01), ncols=4, frameon=False, fontsize=10
    )
    _format_axis(axis)
    return figure
