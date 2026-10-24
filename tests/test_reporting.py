"""Specify data-file persistence inside a caller-supplied working directory.

Complete-run isolation and publication belong to publish_report(), tested later.
Imports stay inside tests so the missing reporting module does not block collection.
"""

import csv
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

METRIC_COLUMNS = [
    "date",
    "table_no",
    "currency",
    "rate",
    "previous_date",
    "previous_rate",
    "absolute_change",
    "percentage_change",
    "indexed_rate",
]
SUMMARY_COLUMNS = ["currency", "finding", *METRIC_COLUMNS[:2], *METRIC_COLUMNS[3:]]


@pytest.fixture
def frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    # Precomputed values are inputs to persistence, not recalculated test oracles.
    metrics = pd.DataFrame(
        {
            "date": ["2025-01-10", "2025-01-13"] * 4,
            "table_no": ["006/A/NBP/2025", "007/A/NBP/2025"] * 4,
            "currency": ["EUR", "EUR", "GBP", "GBP", "JPY", "JPY", "USD", "USD"],
            "rate": [
                4.0,
                6.0,
                4.0,
                4.0,
                0.02412312345678901,
                0.04824624691357802,
                2.0,
                1.0,
            ],
            "previous_date": [None, "2025-01-10"] * 4,
            "previous_rate": [
                None,
                4.0,
                None,
                4.0,
                None,
                0.02412312345678901,
                None,
                2.0,
            ],
            "absolute_change": [
                None,
                2.0,
                None,
                0.0,
                None,
                0.02412312345678901,
                None,
                -1.0,
            ],
            "percentage_change": [
                None,
                50.00000000000001,
                None,
                0.0,
                None,
                100.0,
                None,
                -50.0,
            ],
            "indexed_rate": [
                100.0,
                150.00000000000003,
                100.0,
                100.0,
                100.0,
                200.0,
                100.0,
                50.0,
            ],
        },
        columns=METRIC_COLUMNS,
    ).astype({"table_no": "string", "currency": "string"})
    for column in ["date", "previous_date"]:
        metrics[column] = pd.to_datetime(metrics[column])

    # Literal source-row choices define the supplied sixteen findings.
    selections = [
        ("EUR", "min_rate", 0),
        ("EUR", "max_rate", 1),
        ("EUR", "largest_fall", None),
        ("EUR", "largest_rise", 1),
        ("GBP", "min_rate", 2),
        ("GBP", "max_rate", 2),
        ("GBP", "largest_fall", None),
        ("GBP", "largest_rise", None),
        ("JPY", "min_rate", 4),
        ("JPY", "max_rate", 5),
        ("JPY", "largest_fall", None),
        ("JPY", "largest_rise", 5),
        ("USD", "min_rate", 7),
        ("USD", "max_rate", 6),
        ("USD", "largest_fall", 7),
        ("USD", "largest_rise", None),
    ]
    rows = []
    for currency, finding, position in selections:
        row = metrics.iloc[position].to_dict() if position is not None else {}
        rows.append({**row, "currency": currency, "finding": finding})
    findings = pd.DataFrame(rows, columns=SUMMARY_COLUMNS).astype(
        {**metrics.dtypes.to_dict(), "finding": "string"}
    )
    # Named, non-default indices must not leak into the CSVs.
    metrics.index = pd.RangeIndex(10, 18, name="source_row")
    findings.index = pd.RangeIndex(30, 46, name="finding_row")
    return metrics, findings


@pytest.fixture
def raw_by_window() -> dict[tuple[date, date], bytes]:
    return {
        (date(2025, 1, 10), date(2025, 1, 12)): (
            '\r\n[ { "table": "A", "no": "006/A/NBP/2025", '
            '"effectiveDate": "2025-01-10", "rates": ['
            '{"code": "EUR", "mid": 4.0}, {"code": "GBP", "mid": 4.0}, '
            '{"code": "JPY", "mid": 0.02412312345678901, "currency": "jen japoński"}, '
            '{"code": "USD", "mid": 2.0} ] } ]\n\t'
        ).encode(),
        (date(2025, 1, 13), date(2025, 1, 15)): (
            b'[{"table":"A","no":"007/A/NBP/2025","effectiveDate":"2025-01-13",'
            b'"rates":[{"code":"EUR","mid":6.0},{"code":"GBP","mid":4.0},'
            b'{"code":"JPY","mid":0.04824624691357802},{"code":"USD","mid":1.0}]}]\n'
        ),
    }


def test_csv_files_preserve_supplied_frames_and_leave_inputs_unchanged(
    tmp_path: Path,
    frames: tuple[pd.DataFrame, pd.DataFrame],
    raw_by_window: dict[tuple[date, date], bytes],
) -> None:
    from pln_fx.reporting import write_data_files

    metrics, findings = frames
    before = [frame.copy(deep=True) for frame in frames]
    directory = tmp_path / "nested" / "working"

    result = write_data_files(metrics, findings, raw_by_window, directory)

    assert result is None
    assert {path.name for path in directory.iterdir()} == {
        "rates.csv",
        "summary.csv",
        "raw",
    }
    for filename, expected in [("rates.csv", metrics), ("summary.csv", findings)]:
        path = directory / filename
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            assert reader.fieldnames == expected.columns.tolist()
            records = list(reader)
        for column in expected.columns:
            assert [record[column] == "" for record in records] == expected[
                column
            ].isna().tolist()
        for column in ["date", "previous_date"]:
            assert [record[column] for record in records] == [
                "" if pd.isna(value) else value.strftime("%Y-%m-%d")
                for value in expected[column]
            ]
        actual = pd.read_csv(
            path,
            encoding="utf-8",
            parse_dates=["date", "previous_date"],
        )
        assert_frame_equal(
            actual,
            expected.reset_index(drop=True),
            check_dtype=False,
            check_exact=True,
        )
    for actual, original in zip(frames, before, strict=True):
        assert_frame_equal(actual, original, check_exact=True)


def test_raw_files_preserve_exact_bytes_and_window_association(
    tmp_path: Path,
    frames: tuple[pd.DataFrame, pd.DataFrame],
    raw_by_window: dict[tuple[date, date], bytes],
) -> None:
    from pln_fx.reporting import write_data_files

    before = raw_by_window.copy()

    write_data_files(*frames, raw_by_window, tmp_path / "working")

    raw = tmp_path / "working" / "raw"
    assert {path.name for path in raw.iterdir()} == {
        "table_a_2025-01-10_2025-01-12.json",
        "table_a_2025-01-13_2025-01-15.json",
    }
    assert (raw / "table_a_2025-01-10_2025-01-12.json").read_bytes() == raw_by_window[
        date(2025, 1, 10), date(2025, 1, 12)
    ]
    assert (raw / "table_a_2025-01-13_2025-01-15.json").read_bytes() == raw_by_window[
        date(2025, 1, 13), date(2025, 1, 15)
    ]
    assert raw_by_window == before


def test_repeated_writes_replace_data_files_in_the_same_working_directory(
    tmp_path: Path,
    frames: tuple[pd.DataFrame, pd.DataFrame],
    raw_by_window: dict[tuple[date, date], bytes],
) -> None:
    from pln_fx.reporting import write_data_files

    directory = tmp_path / "working"
    write_data_files(*frames, raw_by_window, directory)
    paths = [path for path in directory.rglob("*") if path.is_file()]
    before = {path.relative_to(directory): path.read_bytes() for path in paths}
    for path in paths:
        path.write_bytes(b"stale contents")

    write_data_files(*frames, raw_by_window, directory)

    after = {
        path.relative_to(directory): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }
    assert after == before


@pytest.mark.parametrize("blocked_location", ["output-root", "rates.csv"])
def test_filesystem_errors_include_the_affected_path(
    tmp_path: Path,
    frames: tuple[pd.DataFrame, pd.DataFrame],
    raw_by_window: dict[tuple[date, date], bytes],
    blocked_location: str,
) -> None:
    from pln_fx.reporting import write_data_files

    directory = tmp_path / "working"
    if blocked_location == "output-root":
        blocked = directory
        blocked.write_text("This is a file, not an output directory.", encoding="utf-8")
    else:
        directory.mkdir()
        blocked = directory / "rates.csv"
        blocked.mkdir()

    with pytest.raises(OSError) as exc_info:
        write_data_files(*frames, raw_by_window, directory)

    assert str(blocked) in str(exc_info.value)


def test_report_formats_supplied_findings_without_mutating_inputs(
    frames: tuple[pd.DataFrame, pd.DataFrame],
    raw_by_window: dict[tuple[date, date], bytes],
) -> None:
    from datetime import UTC, datetime

    from pln_fx.reporting import render_report

    metrics, findings = frames
    # A sentinel distinguishes supplied findings from a recalculation using metrics.
    findings.loc[
        (findings["currency"] == "EUR") & (findings["finding"] == "largest_rise"),
        "percentage_change",
    ] = 12.345
    before = [frame.copy(deep=True) for frame in frames]

    report = render_report(
        metrics,
        findings,
        raw_by_window,
        requested_start=date(2025, 1, 9),
        requested_end=date(2025, 1, 15),
        generated_at=datetime(2026, 10, 24, 11, 15, tzinfo=UTC),
    )

    labels = {
        "Historical minimum",
        "Historical maximum",
        "Largest daily depreciation",
        "Largest daily appreciation",
    }
    rows = [
        [cell.strip() for cell in line.split("|")[1:-1]]
        for line in report.splitlines()
        if line.startswith("|") and line.split("|")[1].strip() in labels
    ]
    assert len(rows) == 16
    for label in labels:
        assert sum(row[0] == label for row in rows) == 4
    for currency in ["EUR", "GBP", "JPY", "USD"]:
        assert f"PLN per 1 {currency}" in report
    assert rows[0] == ["Historical minimum", "2025-01-10", "4", "N/A", "N/A"]
    assert rows[3] == [
        "Largest daily appreciation",
        "2025-01-13",
        "6",
        "2025-01-10 (4)",
        "+2 (+12.35%)",
    ]
    assert rows[8] == ["Historical minimum", "2025-01-10", "0.024123123", "N/A", "N/A"]
    assert rows[11] == [
        "Largest daily appreciation",
        "2025-01-13",
        "0.048246247",
        "2025-01-10 (0.024123123)",
        "+0.024123123 (+100%)",
    ]
    assert rows[14] == [
        "Largest daily depreciation",
        "2025-01-13",
        "1",
        "2025-01-10 (2)",
        "-1 (-50%)",
    ]
    for actual, original in zip(frames, before, strict=True):
        assert_frame_equal(actual, original, check_exact=True)


def test_report_covers_null_events_provenance_and_utc_generation_time(
    frames: tuple[pd.DataFrame, pd.DataFrame],
    raw_by_window: dict[tuple[date, date], bytes],
) -> None:
    from datetime import datetime, timedelta, timezone

    from pln_fx.reporting import render_report

    options = {
        "requested_start": date(2025, 1, 9),
        "requested_end": date(2025, 1, 15),
    }
    before = raw_by_window.copy()
    report = render_report(
        *frames,
        raw_by_window,
        **options,
        generated_at=datetime(
            2026, 10, 24, 13, 15, tzinfo=timezone(timedelta(hours=2))
        ),
    )

    assert "2025-01-09 to 2025-01-15, inclusive." in report
    assert "2025-01-10 to 2025-01-13" in report
    assert "8 across 2 publication dates" in report
    assert "2026-10-24 11:15:00 UTC" in report
    assert report.count("No depreciation observed") == 3
    assert report.count("No appreciation observed") == 2
    for line in report.splitlines():
        if "No depreciation observed" in line or "No appreciation observed" in line:
            cells = [cell.strip() for cell in line.split("|")[1:-1]]
            assert cells[2:] == ["N/A", "N/A", "N/A"]
    for destination in [
        "https://api.nbp.pl/en.html",
        "rates.csv",
        "summary.csv",
        "charts/historical_rates.png",
        "charts/indexed_rates.png",
        "raw/table_a_2025-01-10_2025-01-12.json",
        "raw/table_a_2025-01-13_2025-01-15.json",
        "https://api.nbp.pl/api/exchangerates/tables/a/2025-01-10/2025-01-12/?format=json",
        "https://api.nbp.pl/api/exchangerates/tables/a/2025-01-13/2025-01-15/?format=json",
    ]:
        assert f"]({destination})" in report
    assert raw_by_window == before

    with pytest.raises(ValueError):
        render_report(
            *frames, raw_by_window, **options, generated_at=datetime(2026, 10, 24)
        )
