"""Offline integration of validated analytical outputs and complete publication."""

import re
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib.image as mimage
import pandas as pd
import pytest
from matplotlib.figure import Figure
from pandas.testing import assert_frame_equal

from pln_fx.analysis import calculate_rate_metrics, summarize_rates
from pln_fx.nbp import parse_window
from pln_fx.reporting import publish_report
from pln_fx.transform import to_rates_frame

START = date(2025, 10, 22)
END = date(2025, 10, 23)
ARTIFACTS = {
    "rates.csv",
    "summary.csv",
    "report.md",
    "charts/historical_rates.png",
    "charts/indexed_rates.png",
    "raw/table_a_2025-10-22_2025-10-22.json",
    "raw/table_a_2025-10-23_2025-10-23.json",
}


@pytest.fixture
def analytical_outputs() -> tuple[
    pd.DataFrame, pd.DataFrame, dict[tuple[date, date], bytes]
]:
    first = (
        Path(__file__).parent / "fixtures" / "nbp_table_a_excerpt.json"
    ).read_bytes()
    second = b"""[
  {"table":"A","no":"206/A/NBP/2025","effectiveDate":"2025-10-23","rates":[
    {"code":"EUR","mid":4.2347},{"code":"GBP","mid":4.9746},
    {"code":"JPY","mid":0.023928},{"code":"USD","mid":3.6519}
  ]}
]\n"""
    raw = {(START, START): b"[\n" + first + b"\n]\n", (END, END): second}
    tables = [
        table
        for (start, end), body in raw.items()
        for table in parse_window(body, start, end)
    ]
    metrics = calculate_rate_metrics(to_rates_frame(tables))
    return metrics, summarize_rates(metrics), raw


def file_contents(directory: Path) -> dict[str, bytes]:
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


def test_complete_publication_and_rerun_preserve_previous_results(
    tmp_path: Path,
    analytical_outputs: tuple[
        pd.DataFrame, pd.DataFrame, dict[tuple[date, date], bytes]
    ],
) -> None:
    metrics, findings, raw = analytical_outputs
    before = [frame.copy(deep=True) for frame in [metrics, findings]]
    original_raw = raw.copy()
    root = tmp_path / "nested" / "outputs"

    first = publish_report(
        metrics, findings, raw, root, requested_start=START, requested_end=END
    )

    assert isinstance(first, Path)
    assert first.parent == root and first.name.startswith("run-")
    snapshot = file_contents(first)
    assert set(snapshot) == ARTIFACTS
    for filename, expected in [("rates.csv", metrics), ("summary.csv", findings)]:
        actual = pd.read_csv(
            first / filename, parse_dates=["date", "previous_date"], encoding="utf-8"
        )
        # CSV carries values, not pandas datetime-resolution or string metadata.
        actual = actual.astype(expected.dtypes.to_dict())
        assert_frame_equal(actual, expected.reset_index(drop=True), check_exact=True)
    assert len(pd.read_csv(first / "summary.csv")) == 16
    for (start, end), body in raw.items():
        assert snapshot[f"raw/table_a_{start}_{end}.json"] == body
    for filename in ["historical_rates.png", "indexed_rates.png"]:
        path = first / "charts" / filename
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        image = mimage.imread(path)
        assert image.ndim == 3 and image.shape[0] > 0 and image.shape[1] > 0
    report = (first / "report.md").read_text(encoding="utf-8")
    assert "2025-10-22 to 2025-10-23" in report
    assert "PLN per 1 JPY" in report and "0.024123" in report
    for target in re.findall(r"\]\(([^)]+)\)", report):
        if not target.startswith("https://"):
            assert (first / target).is_file(), target

    second = publish_report(
        metrics, findings, raw, root, requested_start=START, requested_end=END
    )

    assert second != first
    assert set(file_contents(second)) == ARTIFACTS
    assert file_contents(first) == snapshot
    assert set(root.iterdir()) == {first, second}
    for actual, original in zip([metrics, findings], before, strict=True):
        assert_frame_equal(actual, original, check_exact=True)
    assert raw == original_raw


def test_chart_save_failure_cleans_staging_and_releases_figures(
    tmp_path: Path,
    analytical_outputs: tuple[
        pd.DataFrame, pd.DataFrame, dict[tuple[date, date], bytes]
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "outputs"
    previous = publish_report(
        *analytical_outputs, root, requested_start=START, requested_end=END
    )
    snapshot = file_contents(previous)
    original_savefig = Figure.savefig
    figures: list[Figure] = []
    failure = OSError("Simulated indexed PNG save failure")

    def savefig(figure: Figure, path: Path, **kwargs: Any) -> None:
        figures.append(figure)
        if path.name == "indexed_rates.png":
            path.write_bytes(b"partial PNG")
            raise failure
        original_savefig(figure, path, **kwargs)

    monkeypatch.setattr(Figure, "savefig", savefig)

    with pytest.raises(OSError) as exc_info:
        publish_report(
            *analytical_outputs, root, requested_start=START, requested_end=END
        )

    assert exc_info.value is failure
    assert len(figures) == 2
    assert all(figure.get_axes() == [] for figure in figures)
    assert set(root.iterdir()) == {previous}
    assert file_contents(previous) == snapshot


def test_final_rename_failure_cleans_staging_and_preserves_previous_run(
    tmp_path: Path,
    analytical_outputs: tuple[
        pd.DataFrame, pd.DataFrame, dict[tuple[date, date], bytes]
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "outputs"
    previous = publish_report(
        *analytical_outputs, root, requested_start=START, requested_end=END
    )
    snapshot = file_contents(previous)
    failure = OSError("Simulated publication rename failure")

    def fail_rename(staging: Path, destination: Path) -> Path:
        assert set(file_contents(staging)) == ARTIFACTS
        assert destination.parent == root
        raise failure

    monkeypatch.setattr(Path, "rename", fail_rename)

    with pytest.raises(OSError) as exc_info:
        publish_report(
            *analytical_outputs, root, requested_start=START, requested_end=END
        )

    assert exc_info.value is failure
    assert set(root.iterdir()) == {previous}
    assert file_contents(previous) == snapshot
