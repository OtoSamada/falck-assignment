"""Currency selection and tabular dataset validation."""

from datetime import date

import pandas as pd

from pln_fx.nbp import NBPTable

_TARGET_CURRENCIES = {"EUR", "USD", "GBP", "JPY"}


def to_rates_frame(tables: list[NBPTable]) -> pd.DataFrame:
    """Return sorted target rates from validated tables, in PLN per one unit.

    Reject missing target currencies and duplicate observations before filtering.
    """
    if not tables:
        raise ValueError("No NBP tables to transform")

    rows: list[tuple[date, str, str, float]] = []
    for table in tables:
        missing = _TARGET_CURRENCIES - {rate.code for rate in table.rates}
        if missing:
            raise ValueError(
                f"Table {table.no} dated {table.effective_date} "
                f"is missing currencies: {', '.join(sorted(missing))}"
            )
        rows.extend(
            (table.effective_date, table.no, rate.code, rate.mid)
            for rate in table.rates
        )

    frame = pd.DataFrame(rows, columns=["date", "table_no", "currency", "rate"])
    duplicates = frame.duplicated(subset=["currency", "date"])
    if duplicates.any():
        duplicate = frame.loc[duplicates].iloc[0]
        raise ValueError(
            f"Duplicate observation for {duplicate['currency']} on {duplicate['date']}"
        )

    frame = frame.loc[frame["currency"].isin(_TARGET_CURRENCIES)].astype(
        {"table_no": "string", "currency": "string", "rate": "float64"}
    )
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values(["currency", "date"], ignore_index=True)
