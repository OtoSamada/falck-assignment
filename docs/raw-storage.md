# Monthly raw-data storage

## Purpose

The raw-data store keeps the original response bytes returned by the National
Bank of Poland (NBP) Table A API. These responses are reusable pipeline inputs
under `data/raw/`. Retaining them supports repeat runs, offline reporting, and
source traceability without reconstructing JSON from validated models.

The store groups responses by calendar month. Finalized months use names such as
`2026-09.json`. A provisional month includes its covered date range, for example
`2026-10-01_2026-10-22.json`.

Persistent raw data differs from report snapshots. The store can change when a
month gains publications or NBP publishes a correction. Each
`outputs/run-{id}/raw/` directory contains the exact source bodies used for that
published report. Updating `data/raw/` does not modify an earlier report.

## Monthly data lifecycle

The online `run` and `ingest` modes can retrieve data. The offline `report` mode
uses stored files only.

| Stored state | Online behavior | Offline behavior |
| --- | --- | --- |
| Required month is missing | Retrieve and validate it | Fail with instructions to run an online mode |
| Provisional month exists | Retrieve it again through the required or previously stored coverage | Use it only when it covers the request |
| Finalized historical month exists | Reuse it unless `--refresh` applies or another stored variant contains additional dates | Load the valid file with the widest coverage |
| Stored response is invalid | Retrieve a replacement | Fail without changing the store |
| Historical month is refreshed | Accept validated corrections, subject to publication-date protection | Not applicable |

A month remains provisional through its final calendar day. It becomes
historical on the following day, when an online load can save the full-month
response as `YYYY-MM.json`. A narrower request does not reduce coverage already
held in a valid provisional file.

One narrow exception handles an absent terminal month whose requested prefix
contains only weekend days. For example, a request ending on Sunday, February 1
can use valid January data without requesting an unpublished February response.
The check happens before any HTTP request. It does not suppress HTTP status code
404 responses. Missing intermediate months, terminal prefixes that include a
weekday, and terminal months with stored data follow the normal lifecycle. A
period with no publications still fails.

## Validation and historical corrections

Every downloaded or stored body passes the same main checks. `parse_window()`
validates the JSON structure, Table A identifier, fields, rates, and publication
dates. `to_rates_frame()` then checks target-currency coverage and duplicate
currency-date observations. A filename cannot claim coverage beyond its month or
the supplied `as_of` date.

The store records every publication date found in valid stored variants. A
replacement can change rate values because NBP corrections are legitimate, but
it cannot omit a previously known publication date. The update fails before it
replaces stored data if any known date disappears. This rule protects observed
history without claiming that the application can prove NBP returned every
publication that should exist.

## Failure handling and data integrity

Online updates fetch and validate a response before writing it. The writer saves
bytes to a temporary file in the target directory, then uses `os.replace()` to
replace the target atomically. Existing variants are removed only after the
target is ready. Retrieval, validation, write, or replacement failures therefore
leave the previous stored response available. A cleanup failure can leave more
than one valid variant. Offline loading then selects the widest valid coverage,
while online loading can replace the variants during its next refresh.

Updates are atomic per file, not across an entire multi-month request. Months
saved before a later failure remain reusable. The store also makes no power-loss
durability promise because it does not force file and directory data to disk.

## Offline reporting

Offline mode does not create an HTTP client. Each requested month must have a
valid stored response that covers the required end date, apart from the narrow
weekend exception. The loader can read a broader monthly body, but it returns only
publications within the exact requested dates. Missing, invalid, or insufficient
coverage stops report generation.

## Trade-offs and limitations

Monthly persistence reduces repeated downloads and provides stable inputs for
offline reports. It also adds filename, refresh, correction, and cleanup rules.
A direct download-and-report flow would be reasonable for a one-time assignment
with no offline or incremental-use requirement.

The implementation supports one writer per raw-data directory. Atomic file
replacement does not coordinate concurrent updates, so parallel writers require
directory-level locking. Validation also cannot independently confirm upstream
completeness beyond schema, coverage, currency, duplicate, and known-date checks.
