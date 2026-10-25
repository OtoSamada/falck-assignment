"""Offline CLI integration with real pipeline components and synthetic responses."""

import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import requests

from pln_fx.cli import main

ARGS = ["--start-date", "2025-09-29", "--end-date", "2025-10-02"]
URLS = [
    "https://api.nbp.pl/api/exchangerates/tables/a/2025-09-01/2025-09-30/?format=json",
    "https://api.nbp.pl/api/exchangerates/tables/a/2025-10-01/2025-10-31/?format=json",
]
BODIES = (
    b"""[\n{"table":"A","no":"SYNTHETIC/180/A/2025","effectiveDate":"2025-09-01",
"rates":[{"code":"EUR","mid":100.0},{"code":"GBP","mid":100.0},
{"code":"JPY","mid":1.0},{"code":"USD","mid":100.0}]},
{"table":"A","no":"SYNTHETIC/205/A/2025","effectiveDate":"2025-09-30",
"rates":[{"code":"EUR","mid":4.0},{"code":"GBP","mid":5.0},
{"code":"JPY","mid":0.024},{"code":"USD","mid":3.0},{"code":"CHF","mid":4.5}]}]\n""",
    b"""[\n{"table":"A","no":"SYNTHETIC/206/A/2025","effectiveDate":"2025-10-01",
"rates":[{"code":"EUR","mid":3.0},{"code":"GBP","mid":6.0},
{"code":"JPY","mid":0.018},{"code":"USD","mid":3.3},{"code":"CHF","mid":4.6}]},
{"table":"A","no":"SYNTHETIC/228/A/2025","effectiveDate":"2025-10-31",
"rates":[{"code":"EUR","mid":0.5},{"code":"GBP","mid":0.5},
{"code":"JPY","mid":0.001},{"code":"USD","mid":0.5}]}]\n""",
)
CURRENT_BODY = (
    BODIES[0]
    .replace(b"2025-09-01", b"2026-10-22")
    .replace(b"2025-09-30", b"2026-10-23")
)
ARTIFACTS = {
    "rates.csv",
    "summary.csv",
    "report.md",
    "charts/historical_rates.png",
    "charts/indexed_rates.png",
    "raw/table_a_2025-09-01_2025-09-30.json",
    "raw/table_a_2025-10-01_2025-10-31.json",
}


def response(status: int, body: bytes) -> requests.Response:
    result = requests.Response()
    result.status_code = status
    result._content = body
    result._content_consumed = True
    return result


def test_cli_publishes_complete_report_from_two_monthly_responses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    with patch.object(
        requests.Session,
        "send",
        autospec=True,
        side_effect=[response(200, body) for body in BODIES],
    ) as http:
        assert main(ARGS) == 0

    output = capsys.readouterr()
    run = Path(output.out.strip())
    assert run.is_absolute() and run.parent == tmp_path / "outputs"
    assert output.out.splitlines() == [str(run)]
    assert [call.args[1].url for call in http.call_args_list] == URLS
    assert {
        path.relative_to(run).as_posix() for path in run.rglob("*") if path.is_file()
    } == ARTIFACTS
    assert (run / "raw/table_a_2025-09-01_2025-09-30.json").read_bytes() == BODIES[0]
    assert (run / "raw/table_a_2025-10-01_2025-10-31.json").read_bytes() == BODIES[1]
    assert (tmp_path / "data/raw/2025-09.json").read_bytes() == BODIES[0]
    assert (tmp_path / "data/raw/2025-10.json").read_bytes() == BODIES[1]

    rates = pd.read_csv(run / "rates.csv")
    assert list(
        rates[["currency", "date", "rate"]].itertuples(index=False, name=None)
    ) == [
        ("EUR", "2025-09-30", 4.0),
        ("EUR", "2025-10-01", 3.0),
        ("GBP", "2025-09-30", 5.0),
        ("GBP", "2025-10-01", 6.0),
        ("JPY", "2025-09-30", 0.024),
        ("JPY", "2025-10-01", 0.018),
        ("USD", "2025-09-30", 3.0),
        ("USD", "2025-10-01", 3.3),
    ]
    first = rates.groupby("currency", sort=False).head(1)
    assert (
        first[
            ["previous_date", "previous_rate", "absolute_change", "percentage_change"]
        ]
        .isna()
        .all()
        .all()
    )
    assert first["indexed_rate"].tolist() == [100.0] * 4
    findings = pd.read_csv(run / "summary.csv").set_index(["currency", "finding"])
    assert findings.index.tolist() == [
        (currency, finding)
        for currency in ["EUR", "GBP", "JPY", "USD"]
        for finding in ["min_rate", "max_rate", "largest_fall", "largest_rise"]
    ]
    fall = findings.loc["EUR", "largest_fall"]
    assert (
        fall["date"],
        fall["rate"],
        fall["previous_date"],
        fall["previous_rate"],
    ) == ("2025-10-01", 3.0, "2025-09-30", 4.0)
    assert fall["absolute_change"] == -1.0
    assert fall["percentage_change"] == pytest.approx(-25.0)
    assert findings.loc[("EUR", "min_rate"), "date"] == "2025-10-01"
    assert findings.loc[("EUR", "max_rate"), "date"] == "2025-09-30"
    assert findings.loc[("GBP", "largest_rise"), "percentage_change"] == pytest.approx(
        20.0
    )
    assert findings.loc[("JPY", "largest_fall"), "rate"] == 0.018
    assert findings.loc[("JPY", "largest_fall"), "percentage_change"] == pytest.approx(
        -25.0
    )
    assert findings.loc[("USD", "largest_rise"), "percentage_change"] == pytest.approx(
        10.0
    )
    assert findings.loc[("JPY", "largest_rise")].isna().all()
    for filename in ["historical_rates.png", "indexed_rates.png"]:
        assert (run / "charts" / filename).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    report = (run / "report.md").read_text(encoding="utf-8")
    assert "2025-09-29 to 2025-10-02" in report
    assert "2025-09-30 to 2025-10-01" in report
    assert "PLN per 1 JPY" in report
    for target in re.findall(r"\]\(([^)]+)\)", report):
        if not target.startswith("https://"):
            assert (run / target).is_file(), target


def test_cli_terminal_weekend_replays_offline_and_fetches_when_weekday_arrives(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw = tmp_path / "raw"
    outputs = tmp_path / "outputs"
    january = (
        BODIES[0]
        .replace(b"2025-09-01", b"2026-01-29")
        .replace(b"2025-09-30", b"2026-01-30")
    )
    february = b"""[
{"table":"A","no":"SYNTHETIC/2026-02-02","effectiveDate":"2026-02-02",
"rates":[{"code":"EUR","mid":4.1},{"code":"GBP","mid":5.1},
{"code":"JPY","mid":0.025},{"code":"USD","mid":3.1}]}]\n"""
    arguments = [
        "--start-date",
        "2026-01-29",
        "--end-date",
        "2026-02-01",
        "--data-dir",
        str(raw),
        "--output-dir",
        str(outputs),
    ]
    with (
        patch("pln_fx.cli.datetime") as clock,
        patch.object(
            requests.Session,
            "send",
            autospec=True,
            side_effect=[response(200, january), response(404, b"No data")],
        ) as http,
    ):
        clock.now.return_value = datetime(2026, 2, 1, tzinfo=UTC)
        assert main(arguments) == 0
        first = Path(capsys.readouterr().out.strip())
        assert [call.args[1].url for call in http.call_args_list] == [
            "https://api.nbp.pl/api/exchangerates/tables/a/2026-01-01/2026-01-31/?format=json"
        ]
        assert {path.name: path.read_bytes() for path in raw.iterdir()} == {
            "2026-01.json": january
        }
        rates = pd.read_csv(first / "rates.csv")
        assert rates["date"].tolist() == ["2026-01-29", "2026-01-30"] * 4
        assert len(pd.read_csv(first / "summary.csv")) == 16
        assert (
            first / "raw/table_a_2026-01-01_2026-01-31.json"
        ).read_bytes() == january
        report = (first / "report.md").read_text(encoding="utf-8")
        assert "Requested period: 2026-01-29 to 2026-02-01" in report
        assert "Actual publication coverage: 2026-01-29 to 2026-01-30" in report

        http.reset_mock()
        http.side_effect = AssertionError("Offline reporting must not request data")
        assert main(["--mode", "report", *arguments]) == 0
        offline = Path(capsys.readouterr().out.strip())
        http.assert_not_called()
        assert (offline / "rates.csv").read_bytes() == (
            first / "rates.csv"
        ).read_bytes()

        clock.now.return_value = datetime(2026, 2, 2, tzinfo=UTC)
        http.side_effect = [response(200, february)]
        extended = ["2026-02-02" if arg == "2026-02-01" else arg for arg in arguments]
        assert main(extended) == 0
        later = Path(capsys.readouterr().out.strip())
        assert [call.args[1].url for call in http.call_args_list] == [
            "https://api.nbp.pl/api/exchangerates/tables/a/2026-02-01/2026-02-02/?format=json"
        ]
        assert (raw / "2026-02-01_2026-02-02.json").read_bytes() == february
        assert (
            pd.read_csv(later / "rates.csv")["date"].tolist()
            == ["2026-01-29", "2026-01-30", "2026-02-02"] * 4
        )


def test_cli_ingest_updates_raw_store_without_publishing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw = tmp_path / "raw"
    outputs = tmp_path / "outputs"
    with patch.object(
        requests.Session,
        "send",
        autospec=True,
        side_effect=[response(200, body) for body in BODIES],
    ) as http:
        assert (
            main(
                [
                    "--mode",
                    "ingest",
                    *ARGS,
                    "--data-dir",
                    str(raw),
                    "--output-dir",
                    str(outputs),
                ]
            )
            == 0
        )

    output = capsys.readouterr()
    assert output.out == ""
    assert "Ingestion complete" in output.err and str(raw) in output.err
    assert [call.args[1].url for call in http.call_args_list] == URLS
    assert {path.name: path.read_bytes() for path in raw.iterdir()} == {
        "2025-09.json": BODIES[0],
        "2025-10.json": BODIES[1],
    }
    assert not outputs.exists()


def test_cli_report_uses_current_stored_snapshot_without_http(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    stored = raw / "2026-10-01_2026-10-24.json"
    stored.write_bytes(CURRENT_BODY)
    outputs = tmp_path / "outputs"

    with patch.object(requests.Session, "send", autospec=True) as http:
        assert (
            main(
                [
                    "--mode",
                    "report",
                    "--start-date",
                    "2026-10-22",
                    "--end-date",
                    "2026-10-24",
                    "--data-dir",
                    str(raw),
                    "--output-dir",
                    str(outputs),
                ]
            )
            == 0
        )

    http.assert_not_called()
    output = capsys.readouterr()
    run = Path(output.out.strip())
    assert output.out.splitlines() == [str(run)]
    assert run.parent == outputs
    assert (run / "raw/table_a_2026-10-01_2026-10-24.json").read_bytes() == CURRENT_BODY
    assert (
        pd.read_csv(run / "rates.csv")["date"].tolist()
        == [
            "2026-10-22",
            "2026-10-23",
        ]
        * 4
    )


@pytest.mark.parametrize(
    ("filename", "payload"),
    [
        pytest.param("2026-10-01_2026-10-23.json", CURRENT_BODY, id="missing-coverage"),
        pytest.param("2026-10-01_2026-10-24.json", b"not JSON", id="invalid-json"),
    ],
)
def test_cli_report_fails_for_unusable_stored_data_without_http(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    filename: str,
    payload: bytes,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / filename).write_bytes(payload)
    outputs = tmp_path / "outputs"

    with patch.object(requests.Session, "send", autospec=True) as http:
        assert (
            main(
                [
                    "--mode",
                    "report",
                    "--start-date",
                    "2026-10-22",
                    "--end-date",
                    "2026-10-24",
                    "--data-dir",
                    str(raw),
                    "--output-dir",
                    str(outputs),
                ]
            )
            == 1
        )

    http.assert_not_called()
    output = capsys.readouterr()
    assert output.out == ""
    assert "--mode ingest" in output.err
    assert not outputs.exists()


def test_cli_ingestion_failure_preserves_previous_run_and_reports_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "outputs"
    previous = root / "run-previous"
    previous.mkdir(parents=True)
    report = previous / "report.md"
    report.write_bytes(b"Previous published results\n")
    with patch.object(
        requests.Session,
        "send",
        autospec=True,
        side_effect=[response(200, BODIES[0]), response(404, b"No data")],
    ) as http:
        assert (
            main(
                [*ARGS, "--data-dir", str(tmp_path / "raw"), "--output-dir", str(root)]
            )
            == 1
        )

    output = capsys.readouterr()
    assert output.out == ""
    assert "2025-10-01" in output.err and "404" in output.err
    assert [call.args[1].url for call in http.call_args_list] == URLS
    assert set(root.iterdir()) == {previous}
    assert report.read_bytes() == b"Previous published results\n"


def test_cli_analysis_failure_does_not_publish_single_publication(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "outputs"
    with patch.object(
        requests.Session, "send", autospec=True, return_value=response(200, BODIES[0])
    ):
        assert (
            main(
                [
                    "--start-date",
                    "2025-09-30",
                    "--end-date",
                    "2025-09-30",
                    "--data-dir",
                    str(tmp_path / "raw"),
                    "--output-dir",
                    str(root),
                ]
            )
            == 1
        )

    output = capsys.readouterr()
    assert output.out == ""
    assert "observations" in output.err and "EUR" in output.err
    assert not root.exists()


def test_cli_publication_error_reports_failure_without_overwriting_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_file = tmp_path / "outputs"
    output_file.write_bytes(b"Keep this file")
    with patch.object(
        requests.Session,
        "send",
        autospec=True,
        side_effect=[response(200, body) for body in BODIES],
    ):
        assert (
            main(
                [
                    *ARGS,
                    "--data-dir",
                    str(tmp_path / "raw"),
                    "--output-dir",
                    str(output_file),
                ]
            )
            == 1
        )

    output = capsys.readouterr()
    assert output.out == "" and str(output_file) in output.err
    assert output_file.read_bytes() == b"Keep this file"
    assert set(tmp_path.iterdir()) == {output_file, tmp_path / "raw"}


def test_cli_raw_store_lifecycle_reuses_months_and_preserves_previous_reports(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw = tmp_path / "source-data"
    outputs = tmp_path / "reports"

    def run(arguments: list[str]) -> Path:
        assert (
            main([*arguments, "--data-dir", str(raw), "--output-dir", str(outputs)])
            == 0
        )
        printed = capsys.readouterr().out
        directory = Path(printed.strip())
        assert printed.splitlines() == [str(directory)]
        assert directory.is_absolute() and directory.parent == outputs
        return directory

    november = b"""[
{"table":"A","no":"SYNTHETIC/229/A/2025","effectiveDate":"2025-11-03",
"rates":[{"code":"EUR","mid":4.2},{"code":"GBP","mid":5.8},
{"code":"JPY","mid":0.022},{"code":"USD","mid":3.6}]}]\n"""
    corrected = BODIES[0].replace(b'"mid":4.0', b'"mid":8.0')
    with patch.object(requests.Session, "send", autospec=True) as http:
        http.side_effect = [response(200, body) for body in BODIES]
        first = run(ARGS)
        assert [call.args[1].url for call in http.call_args_list] == URLS
        snapshot = {
            path.relative_to(first): path.read_bytes()
            for path in first.rglob("*")
            if path.is_file()
        }

        http.reset_mock()
        http.side_effect = AssertionError("Stored historical months must not use HTTP")
        warm = run(["--mode", "run", *ARGS])
        narrower = run(["--start-date", "2025-09-30", "--end-date", "2025-10-01"])
        http.assert_not_called()
        assert (warm / "rates.csv").read_bytes() == (first / "rates.csv").read_bytes()
        assert (narrower / "rates.csv").read_bytes() == (
            first / "rates.csv"
        ).read_bytes()

        http.reset_mock()
        http.side_effect = [response(200, november)]
        extended = run(["--start-date", "2025-09-29", "--end-date", "2025-11-03"])
        assert [call.args[1].url for call in http.call_args_list] == [
            "https://api.nbp.pl/api/exchangerates/tables/a/2025-11-01/2025-11-30/?format=json"
        ]
        assert (raw / "2025-11.json").read_bytes() == november
        assert (
            extended / "raw/table_a_2025-11-01_2025-11-30.json"
        ).read_bytes() == november

        http.reset_mock()
        http.side_effect = [response(200, corrected), response(200, BODIES[1])]
        refreshed = run([*ARGS, "--refresh"])
        assert [call.args[1].url for call in http.call_args_list] == URLS

    assert len({first, warm, narrower, extended, refreshed}) == 5
    assert (raw / "2025-09.json").read_bytes() == corrected
    assert (
        refreshed / "raw/table_a_2025-09-01_2025-09-30.json"
    ).read_bytes() == corrected
    findings = pd.read_csv(refreshed / "summary.csv").set_index(["currency", "finding"])
    assert findings.loc[("EUR", "max_rate"), "rate"] == 8.0
    assert findings.loc[("EUR", "largest_fall"), "percentage_change"] == pytest.approx(
        -62.5
    )
    assert {
        path.relative_to(first): path.read_bytes()
        for path in first.rglob("*")
        if path.is_file()
    } == snapshot


@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param(["--start-date", "2025-09-30"], id="missing-end"),
        pytest.param(
            ["--start-date", "2025-02-30", "--end-date", "2025-10-01"],
            id="invalid-calendar-date",
        ),
        pytest.param(
            ["--start-date", "20251022", "--end-date", "2025-10-01"],
            id="noncanonical-date",
        ),
        pytest.param(
            ["--start-date", "2025-10-01", "--end-date", "2025-09-30"],
            id="reversed-range",
        ),
        pytest.param(
            ["--start-date", "2025-09-30", "--end-date", "9999-12-31"], id="future-end"
        ),
        pytest.param(["--mode", "report", *ARGS, "--refresh"], id="report-refresh"),
    ],
)
def test_cli_rejects_invalid_arguments_before_network_access(
    arguments: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    with patch.object(requests.Session, "send", autospec=True) as http:
        with pytest.raises(SystemExit) as exc_info:
            main(arguments)
        http.assert_not_called()

    assert exc_info.value.code == 2
    output = capsys.readouterr()
    assert output.out == "" and "error" in output.err.lower()
    assert list(tmp_path.iterdir()) == []


def test_module_help_exits_successfully_without_producing_outputs(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pln_fx", "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0 and result.stderr == ""
    for option in [
        "--start-date",
        "--end-date",
        "--mode",
        "--output-dir",
        "--data-dir",
        "--refresh",
    ]:
        assert option in result.stdout
    assert "inclusive" in result.stdout.lower()
    assert list(tmp_path.iterdir()) == []
