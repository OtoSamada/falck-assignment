"""Command-line arguments and pipeline orchestration."""

import argparse
import logging
import sys
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path

from pln_fx.analysis import calculate_rate_metrics, summarize_rates
from pln_fx.nbp import NBPClient
from pln_fx.raw_store import load_or_fetch_months, load_stored_months
from pln_fx.reporting import publish_report
from pln_fx.transform import to_rates_frame

_LOGGER = logging.getLogger(__name__)


def _parse_date(value: str) -> date:
    """Require a valid calendar date in canonical ISO format."""
    try:
        parsed = date.fromisoformat(value)
        if parsed.isoformat() != value:
            raise ValueError("Noncanonical date")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Use a valid date in YYYY-MM-DD format"
        ) from exc
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pipeline and return its status; argparse handles argument exits."""
    parser = argparse.ArgumentParser(
        prog="python -m pln_fx",
        description=(
            "Ingest and analyze NBP Table A exchange rates for EUR, USD, GBP, "
            "and JPY against PLN."
        ),
        epilog=(
            "Both date boundaries are inclusive. Dates must not be in the future in UTC. "
            "Run and report modes publish a new report directory and print its absolute "
            "path; ingest mode writes only monthly raw data."
        ),
    )
    parser.add_argument(
        "--start-date", required=True, type=_parse_date, help="First date, YYYY-MM-DD"
    )
    parser.add_argument(
        "--end-date", required=True, type=_parse_date, help="Last date, YYYY-MM-DD"
    )
    parser.add_argument(
        "--mode",
        choices=("run", "ingest", "report"),
        default="run",
        help=(
            "run fetches and reports (default); ingest stores raw data only; "
            "report uses stored data without HTTP"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Output root directory, relative to the working directory (default: outputs)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help=(
            "Monthly raw-data store directory, relative to the working directory "
            "(default: data/raw)"
        ),
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Download every requested month again, including stored historical months",
    )
    args = parser.parse_args(argv)
    if args.start_date > args.end_date:
        parser.error("--start-date must be on or before --end-date")
    if args.mode == "report" and args.refresh:
        parser.error("--refresh cannot be used with --mode report")
    today = datetime.now(UTC).date()
    if args.end_date > today:
        parser.error(f"--end-date must be on or before today in UTC ({today})")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        raw_directory = args.data_dir.expanduser().resolve()
        if args.mode == "report":
            tables, raw_by_window = load_stored_months(
                args.start_date,
                args.end_date,
                raw_directory,
                as_of=today,
            )
        else:
            with NBPClient() as client:
                tables, raw_by_window = load_or_fetch_months(
                    client,
                    args.start_date,
                    args.end_date,
                    raw_directory,
                    as_of=today,
                    refresh=args.refresh,
                )
            if args.mode == "ingest":
                print(
                    f"Ingestion complete: {len(tables)} publications available for the "
                    f"requested period in {raw_directory}",
                    file=sys.stderr,
                )
                return 0

        rates = to_rates_frame(tables)
        _LOGGER.info(
            "Analyzing %d observations across %d publications", len(rates), len(tables)
        )
        metrics = calculate_rate_metrics(rates)
        findings = summarize_rates(metrics)
        output_root = args.output_dir.expanduser().resolve()
        _LOGGER.info("Publishing %d findings to %s", len(findings), output_root)
        run_dir = publish_report(
            metrics,
            findings,
            raw_by_window,
            output_root,
            requested_start=args.start_date,
            requested_end=args.end_date,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(run_dir)
    return 0
