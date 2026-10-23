"""Specify the NBP HTTP contract with offline network responses.

fetch_tables(start, end) returns (tables, raw_by_window), where raw_by_window
maps inclusive date pairs to the original response bytes. The existing parser
runs unchanged. Synthetic publication dates exercise request-window boundaries.
"""

import json
import time
from contextlib import nullcontext
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import Mock
from urllib.parse import parse_qs, urlsplit

import pytest
import requests
from requests.exceptions import ConnectionError, ConnectTimeout, ReadTimeout, SSLError

from pln_fx import nbp

START = date(2025, 10, 22)
END = date(2025, 10, 23)


def table_body(publication_date: str = "2025-10-22") -> bytes:
    path = Path(__file__).parent / "fixtures" / "nbp_table_a_excerpt.json"
    table = json.loads(path.read_bytes())
    table["effectiveDate"] = publication_date
    return ("\n" + json.dumps([table], ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )


def response(
    status: int, body: bytes = b"", headers: dict[str, str] | None = None
) -> requests.Response:
    result = requests.Response()
    result.status_code = status
    result._content = body
    result._content_consumed = True
    result.headers.update(headers or {})
    result.close = Mock(wraps=result.close)
    return result


@pytest.fixture(autouse=True)
def http(monkeypatch: pytest.MonkeyPatch) -> Mock:
    send = Mock(side_effect=AssertionError("Unexpected HTTP request"))

    def fake_send(
        session: requests.Session, request: requests.PreparedRequest, **kwargs: Any
    ) -> requests.Response:
        result = send(session, request, **kwargs)
        result.request = request
        result.url = request.url
        return result

    monkeypatch.setattr(requests.Session, "send", fake_send)
    return send


@pytest.fixture(autouse=True)
def sleep(monkeypatch: pytest.MonkeyPatch) -> Mock:
    sleeper = Mock()
    monkeypatch.setattr(time, "sleep", sleeper)
    if hasattr(nbp, "sleep"):
        monkeypatch.setattr(nbp, "sleep", sleeper)
    return sleeper


@pytest.fixture
def session_close(monkeypatch: pytest.MonkeyPatch) -> Mock:
    close = Mock(wraps=requests.Session.close)

    def close_session(session: requests.Session) -> None:
        close(session)

    monkeypatch.setattr(requests.Session, "close", close_session)
    return close


def test_gets_table_a_json_with_timeouts_and_preserves_source_bytes(
    http: Mock, sleep: Mock
) -> None:
    body = table_body()
    http.side_effect = [response(200, body)]

    with nbp.NBPClient() as client:
        tables, raw_by_window = client.fetch_tables(START, END)

    assert len(tables) == 1
    assert isinstance(tables[0], nbp.NBPTable)
    assert tables[0].effective_date == START
    assert raw_by_window == {(START, END): body}
    assert http.call_count == 1

    _, request = http.call_args.args
    url = urlsplit(request.url)
    assert request.method == "GET"
    assert url.scheme == "https"
    assert url.netloc == "api.nbp.pl"
    assert url.path == "/api/exchangerates/tables/a/2025-10-22/2025-10-23/"
    assert parse_qs(url.query) == {"format": ["json"]}
    assert http.call_args.kwargs["timeout"] == (5, 30)
    assert http.call_args.kwargs["allow_redirects"] is False
    assert http.call_args.kwargs["verify"] is not False
    sleep.assert_not_called()


def test_extracts_windows_sequentially_with_one_session(http: Mock) -> None:
    bodies = [table_body(day) for day in ("2025-01-02", "2025-04-01", "2025-06-30")]
    http.side_effect = [response(200, body) for body in bodies]

    with nbp.NBPClient() as client:
        tables, raw_by_window = client.fetch_tables(date(2025, 1, 1), date(2025, 7, 2))

    assert [table.effective_date for table in tables] == [
        date(2025, 1, 2),
        date(2025, 4, 1),
        date(2025, 6, 30),
    ]
    assert [urlsplit(call.args[1].url).path for call in http.call_args_list] == [
        "/api/exchangerates/tables/a/2025-01-01/2025-03-31/",
        "/api/exchangerates/tables/a/2025-04-01/2025-06-29/",
        "/api/exchangerates/tables/a/2025-06-30/2025-07-02/",
    ]
    assert all(
        call.args[0] is http.call_args_list[0].args[0] for call in http.call_args_list
    )
    assert raw_by_window == {
        (date(2025, 1, 1), date(2025, 3, 31)): bodies[0],
        (date(2025, 4, 1), date(2025, 6, 29)): bodies[1],
        (date(2025, 6, 30), date(2025, 7, 2)): bodies[2],
    }


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(ConnectionError("connection reset"), id="connection-error"),
        pytest.param(ConnectTimeout("connection timed out"), id="connect-timeout"),
        pytest.param(ReadTimeout("read timed out"), id="read-timeout"),
        pytest.param(429, id="rate-limited"),
        pytest.param(500, id="server-error"),
        pytest.param(502, id="bad-gateway"),
        pytest.param(503, id="unavailable"),
    ],
)
def test_retries_transient_failures_then_returns_validated_data(
    http: Mock, sleep: Mock, failure: Exception | int
) -> None:
    first = response(failure) if isinstance(failure, int) else failure
    body = table_body()
    http.side_effect = [first, response(200, body)]

    with nbp.NBPClient() as client:
        tables, raw_by_window = client.fetch_tables(START, END)

    assert len(tables) == 1
    assert raw_by_window == {(START, END): body}
    assert http.call_count == 2
    assert http.call_args_list[0].args[1].url == http.call_args_list[1].args[1].url
    sleep.assert_called_once_with(1)


def test_mixed_failures_share_three_total_attempts(http: Mock, sleep: Mock) -> None:
    http.side_effect = [
        ConnectionError("connection reset"),
        response(429),
        response(503),
    ]

    with pytest.raises(RuntimeError) as exc_info, nbp.NBPClient() as client:
        client.fetch_tables(START, END)

    assert http.call_count == 3
    assert [call.args[0] for call in sleep.call_args_list] == [1, 2]
    assert isinstance(exc_info.value.__cause__, requests.RequestException)
    assert START.isoformat() in str(exc_info.value)
    assert END.isoformat() in str(exc_info.value)


def test_timeouts_stop_after_three_total_attempts(http: Mock, sleep: Mock) -> None:
    failures = [ReadTimeout("read timed out") for _ in range(3)]
    http.side_effect = failures

    with pytest.raises(RuntimeError) as exc_info, nbp.NBPClient() as client:
        client.fetch_tables(START, END)

    assert http.call_count == 3
    assert [call.args[0] for call in sleep.call_args_list] == [1, 2]
    assert exc_info.value.__cause__ is failures[-1]


def test_retry_budget_restarts_for_each_window(http: Mock, sleep: Mock) -> None:
    http.side_effect = [
        response(500),
        ReadTimeout("read timed out"),
        response(200, table_body("2025-01-02")),
        response(503),
        response(429),
        response(200, table_body("2025-04-01")),
    ]

    with nbp.NBPClient() as client:
        tables, _ = client.fetch_tables(date(2025, 1, 1), date(2025, 4, 1))

    assert [table.effective_date for table in tables] == [
        date(2025, 1, 2),
        date(2025, 4, 1),
    ]
    assert http.call_count == 6
    assert [call.args[0] for call in sleep.call_args_list] == [1, 2, 1, 2]
    assert [urlsplit(call.args[1].url).path for call in http.call_args_list] == [
        "/api/exchangerates/tables/a/2025-01-01/2025-03-31/",
    ] * 3 + ["/api/exchangerates/tables/a/2025-04-01/2025-04-01/"] * 3


@pytest.mark.parametrize(
    ("status", "header", "expected_delay"),
    [
        pytest.param(429, "7", 7, id="server-requested-delay"),
        pytest.param(503, "3600", 30, id="server-delay-capped-at-30-seconds"),
        pytest.param(429, "invalid", 1, id="invalid-header-falls-back-to-backoff"),
    ],
)
def test_retry_after_seconds_uses_bounded_waits(
    http: Mock, sleep: Mock, status: int, header: str, expected_delay: int
) -> None:
    first = response(status, headers={"Retry-After": header})
    http.side_effect = [first, response(200, table_body())]

    with nbp.NBPClient() as client:
        client.fetch_tables(START, END)

    assert http.call_count == 2
    sleep.assert_called_once_with(expected_delay)


def test_retry_after_http_date_is_capped_at_30_seconds(http: Mock, sleep: Mock) -> None:
    # A far-future date exercises the cap without patching a clock.
    first = response(429, headers={"Retry-After": "Fri, 01 Jan 2100 00:00:00 GMT"})
    http.side_effect = [first, response(200, table_body())]

    with nbp.NBPClient() as client:
        client.fetch_tables(START, END)

    sleep.assert_called_once_with(30)
    assert http.call_count == 2


@pytest.mark.parametrize("status", [400, 403, 404, 302])
def test_nonretryable_statuses_fail_immediately(
    http: Mock, sleep: Mock, status: int
) -> None:
    returned = response(status, headers={"Location": "https://example.invalid/"})
    http.side_effect = [returned]

    with pytest.raises(RuntimeError) as exc_info, nbp.NBPClient() as client:
        client.fetch_tables(START, END)

    assert http.call_count == 1
    assert START.isoformat() in str(exc_info.value)
    assert END.isoformat() in str(exc_info.value)
    sleep.assert_not_called()


def test_tls_errors_fail_immediately_despite_being_connection_errors(
    http: Mock, sleep: Mock
) -> None:
    failure = SSLError("certificate verification failed")
    http.side_effect = [failure]

    with pytest.raises(RuntimeError) as exc_info, nbp.NBPClient() as client:
        client.fetch_tables(START, END)

    assert http.call_count == 1
    assert exc_info.value.__cause__ is failure
    sleep.assert_not_called()


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(b"[]", id="empty-window"),
        pytest.param(b"not json", id="malformed-json"),
        pytest.param(b'[{"table":"A"}]', id="invalid-table-schema"),
    ],
)
def test_invalid_success_responses_fail_without_retrying(
    http: Mock, sleep: Mock, body: bytes
) -> None:
    returned = response(200, body)
    http.side_effect = [returned]

    with pytest.raises(ValueError), nbp.NBPClient() as client:
        client.fetch_tables(START, END)

    assert http.call_count == 1
    sleep.assert_not_called()


def test_validates_each_response_against_its_own_window(
    http: Mock, sleep: Mock
) -> None:
    returned = response(200, table_body("2025-04-01"))
    http.side_effect = [returned]

    with pytest.raises(ValueError), nbp.NBPClient() as client:
        client.fetch_tables(date(2025, 1, 1), date(2025, 7, 2))

    assert http.call_count == 1
    sleep.assert_not_called()


@pytest.mark.parametrize(
    ("status", "body", "failure_type", "attempts", "delays"),
    [
        pytest.param(404, b"", RuntimeError, 1, [], id="later-no-data"),
        pytest.param(200, b"[]", ValueError, 1, [], id="later-empty-window"),
        pytest.param(503, b"", RuntimeError, 3, [1, 2], id="later-retries-exhausted"),
    ],
)
def test_later_window_failure_aborts_without_a_partial_return(
    http: Mock,
    sleep: Mock,
    status: int,
    body: bytes,
    failure_type: type[Exception],
    attempts: int,
    delays: list[int],
) -> None:
    first = response(200, table_body("2025-01-02"))
    failed_responses = [response(status, body) for _ in range(attempts)]
    http.side_effect = [first, *failed_responses]

    with pytest.raises(failure_type), nbp.NBPClient() as client:
        client.fetch_tables(date(2025, 1, 1), date(2025, 7, 2))

    assert [urlsplit(call.args[1].url).path for call in http.call_args_list] == [
        "/api/exchangerates/tables/a/2025-01-01/2025-03-31/"
    ] + ["/api/exchangerates/tables/a/2025-04-01/2025-06-29/"] * attempts
    assert [call.args[0] for call in sleep.call_args_list] == delays


def test_transport_adapter_retries_are_disabled(http: Mock) -> None:
    http.side_effect = [response(200, table_body())]

    with nbp.NBPClient() as client:
        client.fetch_tables(START, END)

        session, request = http.call_args.args
        assert session.get_adapter(request.url).max_retries.total == 0


@pytest.mark.parametrize(
    ("statuses", "body", "error_type"),
    [
        pytest.param([503, 200], None, None, id="success-after-retry"),
        pytest.param([503, 503, 503], b"", RuntimeError, id="retrieval-failure"),
        pytest.param([200], b"[]", ValueError, id="parsing-failure"),
    ],
)
def test_closes_responses_and_session(
    http: Mock,
    session_close: Mock,
    statuses: list[int],
    body: bytes | None,
    error_type: type[Exception] | None,
) -> None:
    body = table_body() if body is None else body
    returned_responses = [response(status, body) for status in statuses]
    http.side_effect = returned_responses
    expected = pytest.raises(error_type) if error_type is not None else nullcontext()

    with expected, nbp.NBPClient() as client:
        client.fetch_tables(START, END)

    for returned in returned_responses:
        returned.close.assert_called()
    session_close.assert_any_call(http.call_args.args[0])


def test_context_manager_closes_session_and_preserves_caller_errors(
    http: Mock, session_close: Mock
) -> None:
    returned = response(200, table_body())
    http.side_effect = [returned]
    failure = RuntimeError("caller stopped")

    with pytest.raises(RuntimeError) as exc_info, nbp.NBPClient() as client:
        client.fetch_tables(START, END)
        raise failure

    assert exc_info.value is failure
    returned.close.assert_called()
    session_close.assert_any_call(http.call_args.args[0])


def test_reversed_dates_fail_before_any_request(http: Mock, sleep: Mock) -> None:
    with pytest.raises(ValueError), nbp.NBPClient() as client:
        client.fetch_tables(END, START)

    http.assert_not_called()
    sleep.assert_not_called()
