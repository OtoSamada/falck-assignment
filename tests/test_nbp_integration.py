"""Offline integration of HTTP windows, response validation, and transformation.

Use reduced Table A publication excerpts and mock only Requests' network boundary.
"""

from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import requests
from pandas.api.types import is_datetime64_dtype
from pandas.testing import assert_frame_equal

from pln_fx.nbp import NBPClient
from pln_fx.transform import to_rates_frame

START = date(2025, 7, 25)
END = date(2025, 10, 23)
EXPECTED_URLS = [
    "https://api.nbp.pl/api/exchangerates/tables/a/2025-07-25/2025-10-22/?format=json",
    "https://api.nbp.pl/api/exchangerates/tables/a/2025-10-23/2025-10-23/?format=json",
]


@pytest.fixture
def bodies() -> tuple[bytes, bytes]:
    path = Path(__file__).parent / "fixtures" / "nbp_table_a_excerpt.json"
    first = b"[\n" + path.read_bytes() + b"\n]\n"
    second = b"""[
  {"table": "A", "no": "206/A/NBP/2025", "effectiveDate": "2025-10-23",
   "rates": [
     {"code": "JPY", "mid": 0.023928},
     {"code": "USD", "mid": 3.6519},
     {"code": "CHF", "mid": 4.5},
     {"code": "GBP", "mid": 4.8745},
     {"code": "EUR", "mid": 4.2347}
   ]}
]
"""
    return first, second


def response(status: int, body: bytes) -> requests.Response:
    result = requests.Response()
    result.status_code = status
    result._content = body
    result._content_consumed = True
    return result


def test_two_window_extraction_produces_exact_rates_and_preserves_raw_bodies(
    bodies: tuple[bytes, bytes],
) -> None:
    with (
        patch.object(
            requests.Session,
            "send",
            autospec=True,
            side_effect=[response(200, body) for body in bodies],
        ) as http,
        NBPClient() as client,
    ):
        tables, raw_by_window = client.fetch_tables(START, END)
        frame = to_rates_frame(tables)

    assert [call.args[1].url for call in http.call_args_list] == EXPECTED_URLS
    assert raw_by_window == {
        (date(2025, 7, 25), date(2025, 10, 22)): bodies[0],
        (date(2025, 10, 23), date(2025, 10, 23)): bodies[1],
    }
    expected = pd.DataFrame(
        [
            ("2025-10-22", "205/A/NBP/2025", "EUR", 4.2438),
            ("2025-10-23", "206/A/NBP/2025", "EUR", 4.2347),
            ("2025-10-22", "205/A/NBP/2025", "GBP", 4.8746),
            ("2025-10-23", "206/A/NBP/2025", "GBP", 4.8745),
            ("2025-10-22", "205/A/NBP/2025", "JPY", 0.024123),
            ("2025-10-23", "206/A/NBP/2025", "JPY", 0.023928),
            ("2025-10-22", "205/A/NBP/2025", "USD", 3.6591),
            ("2025-10-23", "206/A/NBP/2025", "USD", 3.6519),
        ],
        columns=["date", "table_no", "currency", "rate"],
    )
    expected["date"] = pd.to_datetime(expected["date"])

    assert_frame_equal(frame, expected, check_dtype=False, check_exact=True)
    assert is_datetime64_dtype(frame["date"].dtype)
    assert isinstance(frame["table_no"].dtype, pd.StringDtype)
    assert isinstance(frame["currency"].dtype, pd.StringDtype)
    assert frame["rate"].dtype == "float64"


def test_later_window_failure_raises_instead_of_returning_partial_rates(
    bodies: tuple[bytes, bytes],
) -> None:
    with (
        patch.object(
            requests.Session,
            "send",
            autospec=True,
            side_effect=[response(200, bodies[0]), response(404, b"No data")],
        ) as http,
        pytest.raises(RuntimeError) as exc_info,
        NBPClient() as client,
    ):
        tables, _ = client.fetch_tables(START, END)
        to_rates_frame(tables)

    assert [call.args[1].url for call in http.call_args_list] == EXPECTED_URLS
    assert isinstance(exc_info.value.__cause__, requests.HTTPError)
    assert exc_info.value.__cause__.response.status_code == 404
