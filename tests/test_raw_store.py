"""Monthly raw-data lifecycle with real parsing and only HTTP mocked."""

import json
import os
from datetime import date
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import patch

import pytest
import requests

from pln_fx.nbp import NBPClient
from pln_fx.raw_store import load_or_fetch_months


def body(*days: str, eur: float = 4.0) -> bytes:
    tables = [
        {
            "table": "A",
            "no": f"SYNTHETIC/{day}",
            "effectiveDate": day,
            "rates": [
                {"code": "EUR", "mid": eur},
                {"code": "GBP", "mid": 5.0},
                {"code": "JPY", "mid": 0.024123},
                {"code": "USD", "mid": 3.0},
                {"code": "CHF", "mid": 4.2},
            ],
        }
        for day in days
    ]
    return (json.dumps(tables, indent=2) + "\n").encode()


def response(payload: bytes, status: int = 200) -> requests.Response:
    result = requests.Response()
    result.status_code = status
    result._content = payload
    result._content_consumed = True
    return result


def test_historical_cold_warm_subset_and_extension_use_monthly_source_bytes(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "nested" / "raw"
    february = body("2024-02-28", "2024-02-29")
    march = body("2024-03-01", "2024-03-25")
    april = body("2024-04-02")
    with (
        patch.object(
            requests.Session,
            "send",
            autospec=True,
            side_effect=[response(february), response(march), response(april)],
        ) as http,
        NBPClient() as client,
    ):
        tables, evidence = load_or_fetch_months(
            client, date(2024, 2, 28), date(2024, 3, 2), raw, as_of=date(2024, 5, 2)
        )
        assert [table.effective_date for table in tables] == [
            date(2024, 2, 28),
            date(2024, 2, 29),
            date(2024, 3, 1),
        ]
        assert evidence == {
            (date(2024, 2, 1), date(2024, 2, 29)): february,
            (date(2024, 3, 1), date(2024, 3, 31)): march,
        }
        assert http.call_count == 2
        stale = raw / "2024-02-01_2024-02-28.json"
        stale.write_bytes(body("2024-02-28"))

        tables, warm = load_or_fetch_months(
            client, date(2024, 2, 29), date(2024, 3, 1), raw, as_of=date(2024, 5, 2)
        )
        assert [table.effective_date for table in tables] == [
            date(2024, 2, 29),
            date(2024, 3, 1),
        ]
        assert warm == evidence and http.call_count == 2
        assert not stale.exists()

        load_or_fetch_months(
            client, date(2024, 2, 29), date(2024, 4, 2), raw, as_of=date(2024, 5, 2)
        )

    assert [call.args[1].url for call in http.call_args_list] == [
        "https://api.nbp.pl/api/exchangerates/tables/a/2024-02-01/2024-02-29/?format=json",
        "https://api.nbp.pl/api/exchangerates/tables/a/2024-03-01/2024-03-31/?format=json",
        "https://api.nbp.pl/api/exchangerates/tables/a/2024-04-01/2024-04-30/?format=json",
    ]
    assert {path.name: path.read_bytes() for path in raw.iterdir()} == {
        "2024-02.json": february,
        "2024-03.json": march,
        "2024-04.json": april,
    }


def test_current_month_refreshes_provisional_coverage_after_day_rollover(
    tmp_path: Path,
) -> None:
    path = tmp_path / "2026-10-01_2026-10-22.json"
    path.write_bytes(body("2026-10-21"))
    updated = body("2026-10-21", "2026-10-22")
    with (
        patch.object(
            requests.Session,
            "send",
            autospec=True,
            side_effect=[response(updated), response(updated)],
        ) as http,
        NBPClient() as client,
    ):
        for _ in range(2):
            tables, evidence = load_or_fetch_months(
                client,
                date(2026, 10, 21),
                date(2026, 10, 22),
                tmp_path,
                as_of=date(2026, 10, 23),
            )
            assert [table.effective_date for table in tables] == [
                date(2026, 10, 21),
                date(2026, 10, 22),
            ]
            assert evidence == {(date(2026, 10, 1), date(2026, 10, 22)): updated}

    assert http.call_count == 2 and path.read_bytes() == updated


def test_current_month_extends_through_weekend_without_shrinking_for_narrower_request(
    tmp_path: Path,
) -> None:
    old = tmp_path / "2026-10-01_2026-10-22.json"
    old.write_bytes(body("2026-10-21", "2026-10-22"))
    updated = body("2026-10-21", "2026-10-22", "2026-10-23")
    with (
        patch.object(
            requests.Session,
            "send",
            autospec=True,
            side_effect=[response(updated), response(updated)],
        ) as http,
        NBPClient() as client,
    ):
        tables, _ = load_or_fetch_months(
            client,
            date(2026, 10, 21),
            date(2026, 10, 24),
            tmp_path,
            as_of=date(2026, 10, 24),
        )
        assert [table.effective_date for table in tables] == [
            date(2026, 10, 21),
            date(2026, 10, 22),
            date(2026, 10, 23),
        ]
        tables, evidence = load_or_fetch_months(
            client,
            date(2026, 10, 21),
            date(2026, 10, 22),
            tmp_path,
            as_of=date(2026, 10, 24),
        )

    assert [table.effective_date for table in tables] == [
        date(2026, 10, 21),
        date(2026, 10, 22),
    ]
    assert evidence == {(date(2026, 10, 1), date(2026, 10, 24)): updated}
    assert [call.args[1].url for call in http.call_args_list] == [
        "https://api.nbp.pl/api/exchangerates/tables/a/2026-10-01/2026-10-24/?format=json"
    ] * 2
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == {
        "2026-10-01_2026-10-24.json": updated
    }


def test_last_day_stays_partial_until_full_month_is_retrieved_after_month_end(
    tmp_path: Path,
) -> None:
    payload = body("2026-10-30")
    partial = tmp_path / "2026-10-01_2026-10-31.json"
    complete = tmp_path / "2026-10.json"
    with (
        patch.object(
            requests.Session,
            "send",
            autospec=True,
            side_effect=[response(payload), response(payload)],
        ) as http,
        NBPClient() as client,
    ):
        load_or_fetch_months(
            client,
            date(2026, 10, 30),
            date(2026, 10, 31),
            tmp_path,
            as_of=date(2026, 10, 31),
        )
        assert partial.read_bytes() == payload and not complete.exists()
        load_or_fetch_months(
            client,
            date(2026, 10, 30),
            date(2026, 10, 30),
            tmp_path,
            as_of=date(2026, 11, 1),
        )
        assert complete.read_bytes() == payload and not partial.exists()
        load_or_fetch_months(
            client,
            date(2026, 10, 30),
            date(2026, 10, 30),
            tmp_path,
            as_of=date(2026, 11, 1),
        )

    assert [call.args[1].url for call in http.call_args_list] == [
        "https://api.nbp.pl/api/exchangerates/tables/a/2026-10-01/2026-10-31/?format=json"
    ] * 2


def test_refresh_accepts_corrected_rates_and_retains_full_source_coverage(
    tmp_path: Path,
) -> None:
    path = tmp_path / "2026-09.json"
    path.write_bytes(body("2026-09-29", "2026-09-30"))
    corrected = body("2026-09-29", "2026-09-30", eur=4.2)
    with (
        patch.object(
            requests.Session, "send", autospec=True, return_value=response(corrected)
        ) as http,
        NBPClient() as client,
    ):
        tables, evidence = load_or_fetch_months(
            client,
            date(2026, 9, 30),
            date(2026, 9, 30),
            tmp_path,
            as_of=date(2026, 10, 24),
            refresh=True,
        )

    assert http.call_count == 1
    assert [table.effective_date for table in tables] == [date(2026, 9, 30)]
    assert next(rate.mid for rate in tables[0].rates if rate.code == "EUR") == 4.2
    assert evidence == {(date(2026, 9, 1), date(2026, 9, 30)): corrected}
    assert path.read_bytes() == corrected


@pytest.mark.parametrize(
    ("incoming", "status", "error"),
    [
        pytest.param(
            body("2026-10-21", "2026-10-23"), 200, ValueError, id="lost-publication"
        ),
        pytest.param(b"not JSON", 200, ValueError, id="invalid-replacement"),
        pytest.param(b"No data", 404, RuntimeError, id="failed-retrieval"),
    ],
)
def test_failed_update_preserves_previously_stored_response(
    tmp_path: Path, incoming: bytes, status: int, error: type[Exception]
) -> None:
    path = tmp_path / "2026-10-01_2026-10-22.json"
    original = body("2026-10-21", "2026-10-22")
    path.write_bytes(original)
    with (
        patch.object(
            requests.Session,
            "send",
            autospec=True,
            return_value=response(incoming, status),
        ),
        NBPClient() as client,
        pytest.raises(error),
    ):
        load_or_fetch_months(
            client,
            date(2026, 10, 21),
            date(2026, 10, 23),
            tmp_path,
            as_of=date(2026, 10, 23),
        )

    assert {entry.name: entry.read_bytes() for entry in tmp_path.iterdir()} == {
        path.name: original
    }


@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param(b"not JSON", id="invalid-json"),
        pytest.param(
            body("2026-09-30").replace(b'"JPY"', b'"XXX"'), id="missing-target"
        ),
    ],
)
def test_invalid_stored_response_is_repaired_after_content_validation(
    tmp_path: Path, invalid: bytes
) -> None:
    path = tmp_path / "2026-09.json"
    path.write_bytes(invalid)
    replacement = body("2026-09-30")
    with (
        patch.object(
            requests.Session, "send", autospec=True, return_value=response(replacement)
        ) as http,
        NBPClient() as client,
    ):
        tables, _ = load_or_fetch_months(
            client,
            date(2026, 9, 30),
            date(2026, 9, 30),
            tmp_path,
            as_of=date(2026, 10, 24),
        )

    assert http.call_count == 1
    assert [table.effective_date for table in tables] == [date(2026, 9, 30)]
    assert path.read_bytes() == replacement


def test_raw_store_read_permission_error_is_not_treated_as_missing_data(
    tmp_path: Path,
) -> None:
    path = tmp_path / "2026-09.json"
    path.write_bytes(body("2026-09-30"))
    with (
        patch.object(
            Path, "read_bytes", side_effect=PermissionError("Cannot read raw data")
        ),
        patch.object(requests.Session, "send", autospec=True) as http,
        NBPClient() as client,
        pytest.raises(PermissionError),
    ):
        load_or_fetch_months(
            client,
            date(2026, 9, 30),
            date(2026, 9, 30),
            tmp_path,
            as_of=date(2026, 10, 24),
        )

    http.assert_not_called()


def test_atomic_replace_failure_preserves_stored_response_and_removes_temporary_file(
    tmp_path: Path,
) -> None:
    old = tmp_path / "2026-10-01_2026-10-21.json"
    original = body("2026-10-21")
    old.write_bytes(original)
    replacement = body("2026-10-21", "2026-10-22")

    def fail_replace(source: str | Path, destination: str | Path) -> None:
        source, destination = Path(source), Path(destination)
        assert source.parent == destination.parent == tmp_path
        assert source.read_bytes() == replacement
        assert old.read_bytes() == original
        raise OSError("Simulated atomic replacement failure")

    with (
        patch.object(
            requests.Session, "send", autospec=True, return_value=response(replacement)
        ),
        patch.object(os, "replace", side_effect=fail_replace),
        NBPClient() as client,
        pytest.raises(OSError),
    ):
        load_or_fetch_months(
            client,
            date(2026, 10, 21),
            date(2026, 10, 22),
            tmp_path,
            as_of=date(2026, 10, 22),
        )

    assert {entry.name: entry.read_bytes() for entry in tmp_path.iterdir()} == {
        old.name: original
    }


def test_partial_write_failure_preserves_stored_response_and_removes_temporary_file(
    tmp_path: Path,
) -> None:
    old = tmp_path / "2026-10-01_2026-10-21.json"
    original = body("2026-10-21")
    old.write_bytes(original)
    replacement = body("2026-10-21", "2026-10-22")

    def failing_temporary_file(*args: object, **kwargs: object):
        # The raw store owns and closes this real file's context.
        temporary = NamedTemporaryFile(*args, **kwargs)  # noqa: SIM115
        write = temporary.write

        def fail_write(payload: bytes) -> None:
            write(payload[:8])
            temporary.flush()
            raise OSError("Simulated partial temporary-file write failure")

        temporary.write = fail_write
        return temporary

    with (
        patch.object(
            requests.Session, "send", autospec=True, return_value=response(replacement)
        ),
        patch(
            "pln_fx.raw_store.NamedTemporaryFile", side_effect=failing_temporary_file
        ),
        NBPClient() as client,
        pytest.raises(OSError),
    ):
        load_or_fetch_months(
            client,
            date(2026, 10, 21),
            date(2026, 10, 22),
            tmp_path,
            as_of=date(2026, 10, 22),
        )

    assert {entry.name: entry.read_bytes() for entry in tmp_path.iterdir()} == {
        old.name: original
    }


def test_superseded_file_cleanup_failure_recovers_with_widest_valid_coverage(
    tmp_path: Path,
) -> None:
    old = tmp_path / "2026-10-01_2026-10-21.json"
    old.write_bytes(body("2026-10-21"))
    replacement = body("2026-10-21", "2026-10-22")
    target = tmp_path / "2026-10-01_2026-10-22.json"
    original_unlink = Path.unlink

    def fail_old_unlink(path: Path, *args: object, **kwargs: object) -> None:
        if path == old:
            raise PermissionError("Cannot remove superseded partial")
        original_unlink(path, *args, **kwargs)

    with (
        patch.object(
            requests.Session,
            "send",
            autospec=True,
            side_effect=[response(replacement), response(replacement)],
        ) as http,
        NBPClient() as client,
    ):
        with (
            patch.object(Path, "unlink", fail_old_unlink),
            pytest.raises(PermissionError),
        ):
            load_or_fetch_months(
                client,
                date(2026, 10, 21),
                date(2026, 10, 22),
                tmp_path,
                as_of=date(2026, 10, 22),
            )
        assert old.exists() and target.read_bytes() == replacement
        tables, evidence = load_or_fetch_months(
            client,
            date(2026, 10, 21),
            date(2026, 10, 21),
            tmp_path,
            as_of=date(2026, 10, 22),
        )

    assert [table.effective_date for table in tables] == [date(2026, 10, 21)]
    assert evidence == {(date(2026, 10, 1), date(2026, 10, 22)): replacement}
    assert http.call_count == 2
    assert http.call_args.args[1].url.endswith("/2026-10-01/2026-10-22/?format=json")
    assert {entry.name: entry.read_bytes() for entry in tmp_path.iterdir()} == {
        target.name: replacement
    }
