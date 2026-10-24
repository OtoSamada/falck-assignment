"""Analytical reporting, persistence, and complete-run publication."""

from datetime import UTC, date, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import pandas as pd

from pln_fx.charts import plot_historical_rates, plot_indexed_rates


def write_data_files(
    metrics: pd.DataFrame,
    findings: pd.DataFrame,
    raw_by_window: dict[tuple[date, date], bytes],
    directory: Path,
) -> None:
    """Write data into a working directory, replacing matching files.

    Preserve supplied values and ordering; filesystem errors propagate.
    Unrelated files are left untouched. Complete-run publication is separate.
    """
    directory.mkdir(parents=True, exist_ok=True)

    for filename, frame in [("rates.csv", metrics), ("summary.csv", findings)]:
        frame.to_csv(
            directory / filename,
            index=False,
            encoding="utf-8",
            date_format="%Y-%m-%d",
            na_rep="",
            lineterminator="\n",
        )

    raw_directory = directory / "raw"
    raw_directory.mkdir(exist_ok=True)
    for (start, end), body in raw_by_window.items():
        (raw_directory / f"table_a_{start}_{end}.json").write_bytes(body)


def _report_number(value: float, *, precision: int = 8, signed: bool = False) -> str:
    """Format display values without changing stored precision."""
    if pd.isna(value):
        return "N/A"
    sign = "+" if signed and value != 0 else ""
    return format(value, f"{sign}.{precision}g")


def render_report(
    metrics: pd.DataFrame,
    findings: pd.DataFrame,
    raw_by_window: dict[tuple[date, date], bytes],
    *,
    requested_start: date,
    requested_end: date,
    generated_at: datetime,
) -> str:
    """Return Markdown from validated observations and supplied findings.

    Keep inputs unchanged. Require a timezone-aware generation timestamp.
    """
    if generated_at.utcoffset() is None:
        raise ValueError("generated_at must include a timezone")
    generated_utc = generated_at.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    first = metrics["date"].min()
    last = metrics["date"].max()
    labels = {
        "min_rate": "Historical minimum",
        "max_rate": "Historical maximum",
        "largest_fall": "Largest daily depreciation",
        "largest_rise": "Largest daily appreciation",
    }
    lines = [
        "# Exchange rates against PLN",
        "",
        (
            "Published average exchange rates for EUR, USD, GBP, and JPY against the Polish zloty (PLN). "
            "Source: [National Bank of Poland (NBP) Table A API](https://api.nbp.pl/en.html)."
        ),
        "",
        "## Coverage",
        "",
        f"- Requested period: {requested_start} to {requested_end}, inclusive.",
        f"- Actual publication coverage: {first:%Y-%m-%d} to {last:%Y-%m-%d}.",
        f"- Observations: {len(metrics):,} across {metrics['date'].nunique():,} publication dates.",
        f"- Generated: {generated_utc}.",
        "",
        "## Historical findings",
        "",
        (
            "Rates use up to eight significant digits; percentages use up to four. "
            "Display rounding does not affect rankings or the full-precision CSV files. "
            "N/A marks an undefined value or an event that did not occur."
        ),
    ]
    for currency, group in findings.groupby("currency", sort=False):
        lines.extend(
            [
                "",
                f"### {currency}",
                "",
                f"Rates and absolute changes are PLN per 1 {currency}.",
                "",
                "| Finding | Date | Rate | Previous date (rate) | Change (PLN and %) |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for row in group.itertuples(index=False):
            if pd.isna(row.date):
                event = (
                    "No depreciation observed"
                    if row.finding == "largest_fall"
                    else "No appreciation observed"
                )
                lines.append(f"| {labels[row.finding]} | {event} | N/A | N/A | N/A |")
                continue
            previous = "N/A"
            movement = "N/A"
            if pd.notna(row.previous_date):
                previous = f"{row.previous_date:%Y-%m-%d} ({_report_number(row.previous_rate)})"
                movement = (
                    f"{_report_number(row.absolute_change, signed=True)} "
                    f"({_report_number(row.percentage_change, precision=4, signed=True)}%)"
                )
            lines.append(
                f"| {labels[row.finding]} | {row.date:%Y-%m-%d} | {_report_number(row.rate)} "
                f"| {previous} | {movement} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation and limits",
            "",
            (
                "- Increasing quoted rates mean the foreign currency appreciates against PLN; "
                "decreasing rates mean it depreciates."
            ),
            (
                "- Daily changes compare consecutive NBP publications. Weekends and holidays remain "
                "gaps; changes are not adjusted for the number of calendar days."
            ),
            "- The first observation has no previous in-period rate, so its movement is undefined.",
            (
                "- Historical levels and movement rankings cover the analyzed observations only. "
                "Daily extremes use percentage changes, with exact ties resolved by the earliest date. "
                "Zero changes are excluded from appreciation and depreciation events."
            ),
            (
                "- Indexed rates start at 100 for each currency's first observation. "
                "Higher values mean appreciation against PLN."
            ),
            "- Published average rates do not describe intraday movements or executable trading prices.",
            "",
            "## Charts and data",
            "",
            "- [Historical exchange rates](charts/historical_rates.png), with independent y-axis scales",
            "- [Indexed currency performance](charts/indexed_rates.png), starting at 100",
            "- [Observations and calculated metrics](rates.csv)",
            "- [Historical findings with full precision](summary.csv)",
            "",
            "## Source responses",
            "",
            "Each saved JSON file preserves the original response bytes for its inclusive request window.",
            "",
        ]
    )
    for start, end in sorted(raw_by_window):
        filename = f"table_a_{start}_{end}.json"
        url = (
            f"https://api.nbp.pl/api/exchangerates/tables/a/{start}/{end}/?format=json"
        )
        lines.append(
            f"- {start} to {end}: [Original Table A JSON](raw/{filename}); "
            f"[NBP request]({url})"
        )
    return "\n".join(lines) + "\n"


def publish_report(
    metrics: pd.DataFrame,
    findings: pd.DataFrame,
    raw_by_window: dict[tuple[date, date], bytes],
    output_root: Path,
    *,
    requested_start: date,
    requested_end: date,
) -> Path:
    """Publish all reporting artifacts as a new run, or raise without publishing.

    Stage under the output root for a same-filesystem rename. Ordinary failures
    remove staging and preserve previous runs; abrupt process termination can
    leave staging behind.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(UTC)
    destination = output_root / f"run-{generated_at:%Y%m%dT%H%M%S%fZ}-{uuid4().hex}"

    with TemporaryDirectory(prefix=".pending-", dir=output_root) as directory:
        staging = Path(directory)
        write_data_files(metrics, findings, raw_by_window, staging)
        charts = staging / "charts"
        charts.mkdir()
        for plot, filename in [
            (plot_historical_rates, "historical_rates.png"),
            (plot_indexed_rates, "indexed_rates.png"),
        ]:
            figure = plot(metrics)
            try:
                figure.savefig(
                    charts / filename, format="png", dpi=150, facecolor="white"
                )
            finally:
                # Direct Agg figures have no pyplot manager; release their artists.
                figure.clear()

        report = render_report(
            metrics,
            findings,
            raw_by_window,
            requested_start=requested_start,
            requested_end=requested_end,
            generated_at=generated_at,
        )
        (staging / "report.md").write_text(report, encoding="utf-8")
        if destination.exists():
            raise FileExistsError(f"Run directory already exists: {destination}")
        staging.rename(destination)

    return destination
