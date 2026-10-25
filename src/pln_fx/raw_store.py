"""Persist validated monthly NBP responses and select exact date ranges.

Online loading reuses finalized months, refreshes provisional months, and keeps
original response bytes available for offline reporting and source traceability.
"""

import logging
import os
import re
from calendar import monthrange
from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile

from pln_fx.nbp import NBPClient, NBPTable, parse_window
from pln_fx.transform import to_rates_frame

_LOGGER = logging.getLogger(__name__)
# shortcut: one writer per raw-data directory; add locking before concurrent runs.


def _iter_months(start: date, end: date) -> Iterator[tuple[date, date]]:
    first = start.replace(day=1)
    while True:
        last = first.replace(day=monthrange(first.year, first.month)[1])
        yield first, last
        if last >= end:
            return
        first = last + timedelta(days=1)


def _write_atomic(path: Path, body: bytes) -> None:
    """Replace a response only after its temporary file is written completely."""
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.stem}-", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(body)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _validate_range(start: date, end: date, as_of: date) -> None:
    if start > end:
        raise ValueError("start must be on or before end")
    if end > as_of:
        raise ValueError(f"end must be on or before as_of ({as_of})")


def _load_months(
    client: NBPClient | None,
    start: date,
    end: date,
    raw_directory: Path,
    paths: list[Path],
    *,
    as_of: date,
    refresh: bool,
) -> tuple[list[NBPTable], dict[tuple[date, date], bytes]]:
    """Load monthly responses, fetching validated replacements when permitted.

    Offline loading requires valid stored coverage. Online updates reject any
    replacement that drops a known publication date and write the replacement
    before removing older variants. Returned tables match the requested dates,
    while source bodies can cover broader monthly windows.
    """
    selected: list[NBPTable] = []
    raw_by_window: dict[tuple[date, date], bytes] = {}

    for first, last in _iter_months(start, end):
        complete = raw_directory / f"{first:%Y-%m}.json"
        pattern = re.compile(rf"{first:%Y-%m}(?:-01_({first:%Y-%m}-\d{{2}}))?\.json")
        month_paths = [path for path in paths if pattern.fullmatch(path.name)]
        # No NBP publication can exist in an absent terminal weekend-only prefix.
        if (
            not month_paths
            and first == end.replace(day=1)
            and (end - first).days < 2
            and first.weekday() >= 5
            and end.weekday() >= 5
        ):
            _LOGGER.info(
                "Skipping absent weekend-only month prefix: %s to %s", first, end
            )
            continue
        coverage_end = last if last < as_of else min(last, end)
        required_end = min(last, end)
        known_dates: set[date] = set()
        stored_complete: tuple[list[NBPTable], bytes] | None = None
        stored_responses: list[tuple[date, Path, list[NBPTable], bytes]] = []

        for path in month_paths:
            try:
                stored_end = (
                    last
                    if path == complete
                    else date.fromisoformat(path.stem.partition("_")[2])
                )
                if stored_end > min(last, as_of):
                    raise ValueError(
                        "Stored response coverage extends beyond the month or as_of"
                    )
                coverage_end = max(coverage_end, stored_end)
                body = path.read_bytes()
                tables = parse_window(body, first, stored_end)
                # Stored bodies must also satisfy target-currency dataset rules.
                to_rates_frame(tables)
            except ValueError as exc:
                if client is None:
                    raise ValueError(
                        f"Invalid stored NBP response {path}: {exc}. "
                        "Run with --mode ingest or --mode run to replace it."
                    ) from exc
                _LOGGER.warning(
                    "Invalid stored NBP response %s; repairing: %s", path, exc
                )
                continue

            stored_responses.append((stored_end, path, tables, body))
            known_dates.update(table.effective_date for table in tables)
            if path == complete:
                stored_complete = tables, body

        if client is None:
            if not stored_responses:
                raise ValueError(
                    f"No stored NBP response covers {first} to {required_end}. "
                    "Run with --mode ingest or --mode run first."
                )
            stored_end, target, tables, body = max(
                stored_responses,
                key=lambda response: (response[0], response[1] == complete),
            )
            if stored_end < required_end:
                raise ValueError(
                    f"Stored NBP data for {first:%Y-%m} covers only through "
                    f"{stored_end}; requested through {required_end}. "
                    "Run with --mode ingest or --mode run to extend it."
                )
            coverage_end = stored_end
            _LOGGER.info("Loaded stored NBP response: %s", target)
        # Reuse a final file only when it contains all dates seen in stored variants.
        elif (
            stored_complete is not None
            and last < as_of
            and not refresh
            and known_dates.issubset(
                table.effective_date for table in stored_complete[0]
            )
        ):
            tables, body = stored_complete
            target = complete
            _LOGGER.info("Reusing stored NBP response: %s", target)
        else:
            _LOGGER.info("Downloading NBP data for %s to %s", first, coverage_end)
            tables, evidence = client.fetch_tables(first, coverage_end)
            to_rates_frame(tables)
            # Corrections may change values but must not remove known publications.
            missing = known_dates - {table.effective_date for table in tables}
            if missing:
                raise ValueError(
                    f"NBP replacement for {first} to {coverage_end} loses known "
                    f"publications: {', '.join(map(str, sorted(missing)))}"
                )
            body = evidence[first, coverage_end]
            target = (
                complete
                if last < as_of
                else raw_directory / f"{first}_{coverage_end}.json"
            )
            _write_atomic(target, body)
            _LOGGER.info("Stored NBP response: %s", target)

        if client is not None:
            for path in month_paths:
                if path != target:
                    path.unlink()

        selected.extend(
            table for table in tables if start <= table.effective_date <= end
        )
        raw_by_window[first, coverage_end] = body

    if not selected:
        raise ValueError(f"No NBP publications in requested range {start} to {end}")
    return selected, raw_by_window


def load_or_fetch_months(
    client: NBPClient,
    start: date,
    end: date,
    raw_directory: Path,
    *,
    as_of: date,
    refresh: bool = False,
) -> tuple[list[NBPTable], dict[tuple[date, date], bytes]]:
    """Return in-range tables and their original, possibly broader monthly bodies.

    Refresh open months on every access; finalize them only after month end.
    Validate source data before atomic replacement and retain known publications.
    One writer per directory is supported; no power-loss durability is promised.
    """
    _validate_range(start, end, as_of)
    raw_directory.mkdir(parents=True, exist_ok=True)
    paths = sorted(raw_directory.iterdir())
    return _load_months(
        client,
        start,
        end,
        raw_directory,
        paths,
        as_of=as_of,
        refresh=refresh,
    )


def load_stored_months(
    start: date,
    end: date,
    raw_directory: Path,
    *,
    as_of: date,
) -> tuple[list[NBPTable], dict[tuple[date, date], bytes]]:
    """Load validated monthly responses without creating an HTTP client."""
    _validate_range(start, end, as_of)
    try:
        paths = sorted(raw_directory.iterdir())
    except FileNotFoundError as exc:
        raise ValueError(
            f"Raw-data store does not exist: {raw_directory}. "
            "Run with --mode ingest or --mode run first."
        ) from exc
    return _load_months(
        None,
        start,
        end,
        raw_directory,
        paths,
        as_of=as_of,
        refresh=False,
    )
