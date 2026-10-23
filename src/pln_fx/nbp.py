"""NBP HTTP client, date windows, and API response validation."""

import logging
import time
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Literal, Self

import requests
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from requests.adapters import HTTPAdapter

_LOGGER = logging.getLogger(__name__)
_MAX_ATTEMPTS = 3


class NBPRate(BaseModel):
    """Middle rate in PLN per one foreign currency unit, including PLN per 1 JPY."""

    model_config = ConfigDict(strict=True, extra="ignore")

    code: str = Field(pattern=r"^[A-Z]{3}$")
    mid: float = Field(gt=0, allow_inf_nan=False)


class NBPTable(BaseModel):
    """A published NBP Table A with its original rates."""

    model_config = ConfigDict(strict=True, extra="ignore")

    table: Literal["A"]
    no: str = Field(pattern=r"\S")
    effective_date: date = Field(alias="effectiveDate")
    rates: list[NBPRate] = Field(min_length=1)


_TABLES_ADAPTER = TypeAdapter(list[NBPTable])


def iter_date_windows(start: date, end: date) -> Iterator[tuple[date, date]]:
    """Yield consecutive inclusive windows of at most 90 calendar days."""
    if start > end:
        raise ValueError("start must be on or before end")

    while True:
        window_end = start + timedelta(days=min(89, (end - start).days))
        yield start, window_end
        if window_end == end:
            return
        start = window_end + timedelta(days=1)


def parse_window(payload: bytes, start: date, end: date) -> list[NBPTable]:
    """Parse a nonempty Table A array within the inclusive requested window."""
    try:
        tables = _TABLES_ADAPTER.validate_json(payload, strict=True)
    except ValidationError as exc:
        raise ValueError(f"Invalid NBP response for {start} to {end}: {exc}") from exc

    if not tables:
        raise ValueError(f"No NBP tables for {start} to {end}")

    for table in tables:
        if not start <= table.effective_date <= end:
            raise ValueError(
                f"Table {table.no} dated {table.effective_date} "
                f"is outside requested window {start} to {end}"
            )

    return tables


def _retry_delay(attempt: int, retry_after: str | None) -> float:
    """Combine exponential backoff with Retry-After, capped at 30 seconds."""
    backoff = min(30.0, 2.0 ** (attempt - 1))
    if not retry_after:
        return backoff

    server_delay: float
    try:
        server_delay = int(retry_after)
    except ValueError:
        try:
            server_delay = (
                parsedate_to_datetime(retry_after) - datetime.now(UTC)
            ).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return backoff

    return min(30.0, max(backoff, server_delay))


class NBPClient:
    """Fetch validated Table A windows; use as a context manager."""

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.mount("https://", HTTPAdapter(max_retries=0))

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self._session.close()

    def fetch_tables(
        self, start: date, end: date
    ) -> tuple[list[NBPTable], dict[tuple[date, date], bytes]]:
        """Return all validated tables and source bodies, or raise on failure."""
        tables: list[NBPTable] = []
        raw_by_window: dict[tuple[date, date], bytes] = {}

        for window_start, window_end in iter_date_windows(start, end):
            _LOGGER.info("Fetching NBP Table A for %s to %s", window_start, window_end)
            try:
                body = self._request_window(window_start, window_end)
                window_tables = parse_window(body, window_start, window_end)
            except (ValueError, RuntimeError) as exc:
                _LOGGER.error(
                    "NBP extraction failed for %s to %s: %s",
                    window_start,
                    window_end,
                    exc,
                )
                raise

            tables.extend(window_tables)
            raw_by_window[window_start, window_end] = body
            _LOGGER.info(
                "Validated %d tables and %d rates for %s to %s",
                len(window_tables),
                sum(len(table.rates) for table in window_tables),
                window_start,
                window_end,
            )

        return tables, raw_by_window

    def _request_window(self, start: date, end: date) -> bytes:
        url = f"https://api.nbp.pl/api/exchangerates/tables/a/{start}/{end}/"
        attempt = 1
        while True:
            try:
                with self._session.get(
                    url,
                    params={"format": "json"},
                    timeout=(5, 30),
                    allow_redirects=False,
                ) as response:
                    if response.status_code != 200:
                        response.raise_for_status()
                        raise requests.HTTPError(
                            f"Unexpected HTTP {response.status_code} from {url}",
                            response=response,
                        )
                    return response.content
            except requests.RequestException as exc:
                error_response = exc.response
                status = (
                    error_response.status_code if error_response is not None else None
                )
                retryable = not isinstance(exc, requests.exceptions.SSLError) and (
                    isinstance(exc, (requests.ConnectionError, requests.Timeout))
                    or status == 429
                    or (status is not None and 500 <= status < 600)
                )
                if not retryable or attempt >= _MAX_ATTEMPTS:
                    raise RuntimeError(
                        f"NBP window {start} to {end} failed after "
                        f"{attempt} attempt(s): {exc}"
                    ) from exc

                retry_after = (
                    error_response.headers.get("Retry-After")
                    if error_response is not None
                    else None
                )
                delay = _retry_delay(attempt, retry_after)
                _LOGGER.warning(
                    "NBP window %s to %s: attempt %d/%d failed (%s); retrying in %s s",
                    start,
                    end,
                    attempt,
                    _MAX_ATTEMPTS,
                    exc,
                    delay,
                )
                time.sleep(delay)
                attempt += 1
