# PLN exchange rate analysis

Falck take-home assignment analyzing five years of EUR, USD, GBP, and JPY exchange rates against the Polish zloty (PLN), using published averages from the [National Bank of Poland (NBP) Table A API](https://api.nbp.pl/en.html).

For each currency, it reports historical minimum and maximum rates and largest daily depreciation and appreciation. Deliverables include CSVs, two charts, Markdown findings, and source responses. Examples use October 23, 2021 to October 22, 2026, inclusive.

## Quick start

Install Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/). From the repository root:

```bash
uv sync --locked
uv run python -m pln_fx --mode run \
  --start-date 2021-10-23 --end-date 2026-10-22
```

Missing or refreshed data requires network access. Reporting commands print the completed run's absolute path to stdout; logs and errors use stderr.

Choose an execution mode:

| Mode | Source data | Result |
| --- | --- | --- |
| `run` (default) | Reuses or retrieves monthly responses | Analyzes and publishes a report |
| `ingest` | Updates the monthly raw-data store | Saves source data only |
| `report` | Reads stored responses without HTTP requests | Analyzes and publishes a report |

Retrieve data first, then generate a report offline:

```bash
uv run python -m pln_fx --mode ingest \
  --start-date 2021-10-23 --end-date 2026-10-22
uv run python -m pln_fx --mode report \
  --start-date 2021-10-23 --end-date 2026-10-22
```

`--start-date` and `--end-date` are required, inclusive `YYYY-MM-DD` dates, ordered and not in the future in UTC. `--data-dir` defaults to `data/raw/`; `--output-dir` defaults to `outputs/`. Relative paths use the working directory. `--refresh` downloads every requested month again in `run` or `ingest`; `report` rejects it. Use `--help` for details.

## Architecture and source reuse

Modules under `src/pln_fx/` separate API retrieval, monthly storage, validation and transformation, Pandas analysis, and report publication. Requests and Pydantic handle the API boundary; Matplotlib produces PNGs.

`data/raw/` is reusable, mutable storage. Historical months use `YYYY-MM.json`; provisional months use `YYYY-MM-01_YYYY-MM-DD.json`. Ingestion refreshes the current month on access and finalizes it after month-end. Historical months are reused until `--refresh` retrieves corrections. Atomic replacements can change rates but cannot remove known publication dates.

`outputs/run-{id}/` contains independent snapshots. Artifacts are staged together, then published by directory rename. Ordinary failures remove staging and preserve previous runs. Months stored before a later failure remain available. Raw data and outputs are excluded from Git.

## Data Contracts & Quality Guarantees

API responses must be nonempty Table A JSON arrays. Two Pydantic models require table identifiers, numbers, publication dates, uppercase currency codes, and positive, finite numeric rates. Unrelated fields and valid non-target currencies are accepted.

Downloaded and stored responses undergo schema, request-window, currency-coverage, and duplicate checks. Each publication must contain all four target currencies exactly once; currency-date pairs must be unique. Analysis requires at least two observations per currency and finite calculated metrics.

The analytical contracts are:

| Output | Granularity | Columns in order |
| --- | --- | --- |
| `rates.csv` | One currency-date observation | `date`, `table_no`, `currency`, `rate`, `previous_date`, `previous_rate`, `absolute_change`, `percentage_change`, `indexed_rate` |
| `summary.csv` | Four findings per currency, 16 rows | `currency`, `finding`, then the observation columns above except `currency` |

Dates use Pandas datetime types, identifiers use strings, and numbers use `float64`. CSV dates use `YYYY-MM-DD`; nulls are empty fields. Rates sort by currency and date. Summaries preserve observation context in this order: `min_rate`, `max_rate`, `largest_fall`, `largest_rise` per currency.

Rates and absolute changes are PLN per one foreign currency unit, including PLN per 1 JPY, without rescaling. First observations have null previous values and changes. Absent rises or falls retain finding rows with null observation fields.

Invalid stored responses trigger replacement attempts in `run` and `ingest`; `report` fails. Invalid downloads, missing currencies, duplicates, uncovered months, or insufficient observations stop reporting without partial publication.

## Analytical methodology

Changes compare consecutive publications within each currency, including across months. Weekends and holidays remain gaps; the first in-range movement is undefined.

For current rate `r`, previous rate `p`, and first in-range rate `r0`:

- Absolute change: `r - p`.
- Percentage change: `100 * (r / p - 1)`.
- Base-100 index: `100 * r / r0`.

Rising rates mean foreign-currency appreciation against PLN; falling rates mean depreciation. Historical extrema use rate levels; daily extremes use percentage changes and exclude zero movements. All exact ties use the earliest date. Rounding applies only to report display.

## Published outputs

Each reporting run contains:

```text
outputs/run-{id}/
├── rates.csv
├── summary.csv
├── report.md
├── charts/
│   ├── historical_rates.png
│   └── indexed_rates.png
└── raw/
    └── table_a_{start}_{end}.json
```

Open `report.md` in a Markdown viewer for findings, requested and actual coverage, generation time, and artifact links. Historical charts use four independent y-axis scales; indexed currencies start at 100. JSON files preserve original bytes for each source window, which may extend beyond the analyzed dates.

## Docker

With Docker running, build and mount persistent directories from the repository root:

```bash
docker build --pull -t falck-pln-fx .
mkdir -p data/raw outputs
docker run --rm \
  -v "$PWD/data/raw:/app/data/raw" \
  -v "$PWD/outputs:/app/outputs" \
  falck-pln-fx --mode run \
  --start-date 2021-10-23 --end-date 2026-10-22
```

For offline reporting from existing data:

```bash
docker run --rm --network none \
  -v "$PWD/data/raw:/app/data/raw:ro" \
  -v "$PWD/outputs:/app/outputs" \
  falck-pln-fx --mode report \
  --start-date 2021-10-23 --end-date 2026-10-22
```

`/app/outputs/` maps to host `outputs/`. The image uses locked dependencies, headless plotting, and user ID `10001`; native Linux mounts need suitable write permissions.

## Tests and CI

Run the automated checks locally:

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

The 193 tests use synthetic fixtures, mocked HTTP, and temporary directories, without live NBP access or local market data. GitHub Actions runs these checks on Ubuntu with Python 3.12 and locked dependencies for branch pushes and pull requests. It does not build or publish Docker images.

## Assumptions and limitations

NBP averages do not describe intraday movements or executable trading prices. Actual coverage can differ from requested boundaries. Connection failures, timeouts, HTTP 429, and HTTP 5xx receive at most three attempts with capped waits.

Use one writer per raw-data directory. Historical corrections can change later results; published runs retain their source bytes. Validation cannot independently prove NBP returned every expected publication. Abrupt termination can leave staging directories behind.
