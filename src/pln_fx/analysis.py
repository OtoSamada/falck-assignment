"""Exchange rate changes and summary calculations."""

from math import isfinite

import pandas as pd


def calculate_rate_metrics(rates: pd.DataFrame) -> pd.DataFrame:
    """Add publication changes and a base-100 index without changing the input.

    Rates remain in PLN per one foreign currency unit.
    Changes compare consecutive publications; the first change is undefined.
    """
    if rates.empty:
        raise ValueError("No exchange-rate observations to analyze")

    metrics = rates.sort_values(["currency", "date"], ignore_index=True).copy()
    grouped = metrics.groupby("currency", sort=False)
    counts = grouped.size()
    insufficient = counts.index[counts < 2].tolist()
    if insufficient:
        raise ValueError(
            f"At least two observations are required per currency: "
            f"{', '.join(insufficient)}"
        )

    metrics[["previous_date", "previous_rate"]] = grouped[["date", "rate"]].shift()
    metrics["absolute_change"] = metrics["rate"] - metrics["previous_rate"]
    metrics["percentage_change"] = (
        metrics["rate"] / metrics["previous_rate"] - 1
    ) * 100
    first_rate = grouped["rate"].transform("first")
    metrics["indexed_rate"] = (metrics["rate"] / first_rate) * 100

    changes = metrics.loc[
        grouped.cumcount() > 0, ["absolute_change", "percentage_change"]
    ]
    if not changes.map(isfinite).all().all():
        raise ValueError("Calculated exchange-rate changes must be finite")
    if not metrics["indexed_rate"].map(isfinite).all():
        raise ValueError("Calculated indexed rates must be finite")

    return metrics


def summarize_rates(metrics: pd.DataFrame) -> pd.DataFrame:
    """Select four findings per currency from precomputed publication metrics.

    Exact ties use the earliest date. Missing directions retain a null finding.
    Preserve each selected observation's context without recalculating metrics.
    """
    columns = [
        "currency",
        "finding",
        "date",
        "table_no",
        "rate",
        "previous_date",
        "previous_rate",
        "absolute_change",
        "percentage_change",
        "indexed_rate",
    ]
    ordered = metrics.sort_values(["currency", "date"], ignore_index=True)
    rows: list[dict[str, object]] = []
    for currency, group in ordered.groupby("currency", sort=False):
        falls = group.loc[group["percentage_change"] < 0]
        rises = group.loc[group["percentage_change"] > 0]
        selections = {
            "min_rate": group["rate"].idxmin(),
            "max_rate": group["rate"].idxmax(),
            "largest_fall": (
                falls["percentage_change"].idxmin() if not falls.empty else None
            ),
            "largest_rise": (
                rises["percentage_change"].idxmax() if not rises.empty else None
            ),
        }
        for finding, index in selections.items():
            row = group.loc[index].to_dict() if index is not None else {}
            row.update(currency=currency, finding=finding)
            rows.append(row)

    return pd.DataFrame(rows, columns=columns).astype(
        {**metrics.dtypes.to_dict(), "finding": "string"}
    )
