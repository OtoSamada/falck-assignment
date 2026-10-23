"""Specify NBP response schemas and parsing, without dataset-level checks.

The JSON fixture is a five-rate excerpt, not a complete publication:
https://api.nbp.pl/api/exchangerates/tables/a/2025-10-22/?format=json
Currency coverage and duplicate detection belong to later dataset validation.
"""

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from pln_fx import nbp

WINDOW_START = date(2025, 10, 22)
WINDOW_END = date(2025, 10, 23)


def encode_json(value: object) -> bytes:
    return json.dumps(value, allow_nan=False).encode("utf-8")


@pytest.fixture
def table_payload() -> dict[str, Any]:
    path = Path(__file__).parent / "fixtures" / "nbp_table_a_excerpt.json"
    return json.loads(path.read_bytes())


@pytest.mark.parametrize(
    ("code", "mid"),
    [
        pytest.param("EUR", 4.2438, id="positive-float"),
        pytest.param("USD", 1, id="positive-json-integer"),
        pytest.param("JPY", 0.024123, id="pln-per-1-jpy-unchanged"),
        pytest.param("CHF", 4.5963, id="non-target-currency"),
        pytest.param("XDR", 4.9854, id="another-valid-currency-code"),
    ],
)
def test_rate_accepts_valid_codes_and_preserves_values(code: str, mid: float) -> None:
    rate = nbp.NBPRate.model_validate_json(encode_json({"code": code, "mid": mid}))

    assert rate.code == code
    assert rate.mid == mid


@pytest.mark.parametrize("field", ["code", "mid"])
def test_rate_requires_each_field(field: str) -> None:
    payload = {"code": "EUR", "mid": 4.2438}
    del payload[field]

    with pytest.raises(ValidationError) as exc_info:
        nbp.NBPRate.model_validate_json(encode_json(payload))

    assert (field,) in [error["loc"] for error in exc_info.value.errors()]


@pytest.mark.parametrize(
    "mid",
    [
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-infinity"),
        pytest.param(float("-inf"), id="negative-infinity"),
        pytest.param(True, id="true"),
        pytest.param(False, id="false"),
        pytest.param("4.2438", id="numeric-string"),
        pytest.param(None, id="null"),
    ],
)
def test_rate_rejects_nonpositive_nonfinite_and_nonnumeric_values(mid: object) -> None:
    # Use Python input so NaN and infinities reach field validation, not JSON syntax.
    with pytest.raises(ValidationError) as exc_info:
        nbp.NBPRate.model_validate({"code": "EUR", "mid": mid})

    assert ("mid",) in [error["loc"] for error in exc_info.value.errors()]


@pytest.mark.parametrize(
    "code",
    [
        pytest.param("", id="empty"),
        pytest.param("eur", id="lowercase"),
        pytest.param("EU", id="too-short"),
        pytest.param("EURO", id="too-long"),
        pytest.param("E1R", id="contains-digit"),
        pytest.param("ÉUR", id="non-ascii"),
        pytest.param(" EUR", id="leading-whitespace"),
        pytest.param(None, id="null"),
        pytest.param(123, id="number"),
    ],
)
def test_rate_rejects_malformed_currency_codes(code: object) -> None:
    with pytest.raises(ValidationError) as exc_info:
        nbp.NBPRate.model_validate_json(encode_json({"code": code, "mid": 4.2438}))

    assert ("code",) in [error["loc"] for error in exc_info.value.errors()]


def test_rate_ignores_unrelated_fields() -> None:
    rate = nbp.NBPRate.model_validate_json(
        encode_json({"code": "EUR", "mid": 4.2438, "currency": "euro", "extra": 123})
    )

    assert rate.model_dump() == {"code": "EUR", "mid": 4.2438}


def test_table_parses_documented_fields_and_nested_rates(
    table_payload: dict[str, Any],
) -> None:
    table = nbp.NBPTable.model_validate_json(encode_json(table_payload))

    assert table.table == "A"
    assert table.no == "205/A/NBP/2025"
    assert table.effective_date == date(2025, 10, 22)
    assert all(isinstance(rate, nbp.NBPRate) for rate in table.rates)
    assert [(rate.code, rate.mid) for rate in table.rates] == [
        ("EUR", 4.2438),
        ("USD", 3.6591),
        ("GBP", 4.8746),
        ("JPY", 0.024123),
        ("CHF", 4.5963),
    ]


@pytest.mark.parametrize("field", ["table", "no", "effectiveDate", "rates"])
def test_table_requires_each_field(table_payload: dict[str, Any], field: str) -> None:
    del table_payload[field]

    with pytest.raises(ValidationError) as exc_info:
        nbp.NBPTable.model_validate_json(encode_json(table_payload))

    assert (field,) in [error["loc"] for error in exc_info.value.errors()]


@pytest.mark.parametrize("table_type", ["B", "C", "a", None])
def test_table_requires_table_a(
    table_payload: dict[str, Any], table_type: object
) -> None:
    table_payload["table"] = table_type

    with pytest.raises(ValidationError):
        nbp.NBPTable.model_validate_json(encode_json(table_payload))


@pytest.mark.parametrize("table_no", ["", "   ", None, 123])
def test_table_requires_a_nonblank_string_number(
    table_payload: dict[str, Any], table_no: object
) -> None:
    table_payload["no"] = table_no

    with pytest.raises(ValidationError):
        nbp.NBPTable.model_validate_json(encode_json(table_payload))


@pytest.mark.parametrize(
    "publication_date",
    [
        pytest.param("2025-02-29", id="invalid-leap-day"),
        pytest.param("2025-13-01", id="invalid-month"),
        pytest.param("22-10-2025", id="wrong-format"),
        pytest.param("2025-10-22T00:00:00", id="datetime-string"),
        pytest.param(1761091200, id="numeric-timestamp"),
        pytest.param(True, id="boolean"),
        pytest.param(None, id="null"),
    ],
)
def test_table_rejects_invalid_publication_dates(
    table_payload: dict[str, Any], publication_date: object
) -> None:
    table_payload["effectiveDate"] = publication_date

    with pytest.raises(ValidationError):
        nbp.NBPTable.model_validate_json(encode_json(table_payload))


def test_table_accepts_a_valid_leap_day(table_payload: dict[str, Any]) -> None:
    table_payload["effectiveDate"] = "2024-02-29"

    table = nbp.NBPTable.model_validate_json(encode_json(table_payload))

    assert table.effective_date == date(2024, 2, 29)


@pytest.mark.parametrize("rates", [[], {}, None, [None]])
def test_table_requires_a_nonempty_list_of_rate_objects(
    table_payload: dict[str, Any], rates: object
) -> None:
    table_payload["rates"] = rates

    with pytest.raises(ValidationError):
        nbp.NBPTable.model_validate_json(encode_json(table_payload))


def test_table_ignores_unrelated_fields(table_payload: dict[str, Any]) -> None:
    table_payload["unrelated"] = {"anything": True}

    table = nbp.NBPTable.model_validate_json(encode_json(table_payload))

    assert "unrelated" not in table.model_dump()


def test_parse_window_accepts_a_single_publication(
    table_payload: dict[str, Any],
) -> None:
    tables = nbp.parse_window(encode_json([table_payload]), WINDOW_START, WINDOW_START)

    assert isinstance(tables, list)
    assert len(tables) == 1
    assert isinstance(tables[0], nbp.NBPTable)
    assert tables[0].no == "205/A/NBP/2025"
    assert tables[0].effective_date == WINDOW_START
    assert [(rate.code, rate.mid) for rate in tables[0].rates] == [
        ("EUR", 4.2438),
        ("USD", 3.6591),
        ("GBP", 4.8746),
        ("JPY", 0.024123),
        ("CHF", 4.5963),
    ]


def test_parse_window_accepts_both_requested_boundaries(
    table_payload: dict[str, Any],
) -> None:
    last_table = {
        **table_payload,
        "no": "206/A/NBP/2025",
        "effectiveDate": "2025-10-23",
    }

    tables = nbp.parse_window(
        encode_json([table_payload, last_table]), WINDOW_START, WINDOW_END
    )

    assert [table.effective_date for table in tables] == [WINDOW_START, WINDOW_END]


@pytest.mark.parametrize("publication_date", ["2025-10-21", "2025-10-24"])
def test_parse_window_rejects_dates_outside_the_requested_window(
    table_payload: dict[str, Any], publication_date: str
) -> None:
    table_payload["effectiveDate"] = publication_date

    with pytest.raises(ValueError):
        nbp.parse_window(encode_json([table_payload]), WINDOW_START, WINDOW_END)


@pytest.mark.parametrize("payload", [b"", b"[", b"<html>Bad Gateway</html>"])
def test_parse_window_rejects_invalid_json(payload: bytes) -> None:
    with pytest.raises(ValueError):
        nbp.parse_window(payload, WINDOW_START, WINDOW_END)


@pytest.mark.parametrize("payload", [None, {}, "text", 123, [None], [123]])
def test_parse_window_requires_an_array_of_table_objects(payload: object) -> None:
    with pytest.raises(ValueError):
        nbp.parse_window(encode_json(payload), WINDOW_START, WINDOW_END)


def test_parse_window_rejects_a_bare_table_object(
    table_payload: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        nbp.parse_window(encode_json(table_payload), WINDOW_START, WINDOW_END)


def test_parse_window_rejects_an_empty_array() -> None:
    with pytest.raises(ValueError):
        nbp.parse_window(b"[]", WINDOW_START, WINDOW_END)


@pytest.mark.parametrize("field", ["code", "mid"])
def test_parse_window_validates_required_nested_rate_fields(
    table_payload: dict[str, Any], field: str
) -> None:
    del table_payload["rates"][0][field]

    with pytest.raises(ValueError):
        nbp.parse_window(encode_json([table_payload]), WINDOW_START, WINDOW_END)


@pytest.mark.parametrize(
    ("field", "value"),
    [("code", "chf"), ("mid", 0), ("mid", True), ("mid", "4.5963")],
)
def test_parse_window_validates_non_target_rates_before_filtering(
    table_payload: dict[str, Any], field: str, value: object
) -> None:
    rate = next(rate for rate in table_payload["rates"] if rate["code"] == "CHF")
    rate[field] = value

    with pytest.raises(ValueError):
        nbp.parse_window(encode_json([table_payload]), WINDOW_START, WINDOW_END)


def test_parse_window_rejects_a_json_number_that_overflows_to_infinity() -> None:
    payload = (
        b'[{"table":"A","no":"205/A/NBP/2025",'
        b'"effectiveDate":"2025-10-22",'
        b'"rates":[{"code":"JPY","mid":1e400}]}]'
    )

    with pytest.raises(ValueError):
        nbp.parse_window(payload, WINDOW_START, WINDOW_END)


def test_parse_window_rejects_a_valid_table_followed_by_an_invalid_table(
    table_payload: dict[str, Any],
) -> None:
    invalid_table = {**table_payload, "effectiveDate": "2025-10-23"}
    del invalid_table["no"]

    with pytest.raises(ValueError):
        nbp.parse_window(
            encode_json([table_payload, invalid_table]), WINDOW_START, WINDOW_END
        )


def test_parse_window_leaves_target_coverage_to_dataset_validation(
    table_payload: dict[str, Any],
) -> None:
    table_payload["rates"] = [
        rate for rate in table_payload["rates"] if rate["code"] == "CHF"
    ]

    tables = nbp.parse_window(encode_json([table_payload]), WINDOW_START, WINDOW_END)

    assert len(tables) == 1
    assert [(rate.code, rate.mid) for rate in tables[0].rates] == [("CHF", 4.5963)]


def test_parse_window_leaves_duplicate_detection_to_dataset_validation(
    table_payload: dict[str, Any],
) -> None:
    tables = nbp.parse_window(
        encode_json([table_payload, table_payload]), WINDOW_START, WINDOW_END
    )

    assert len(tables) == 2
    assert [table.no for table in tables] == ["205/A/NBP/2025", "205/A/NBP/2025"]
