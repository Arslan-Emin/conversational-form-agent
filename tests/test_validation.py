"""Unit tests for deterministic validation. `now` is pinned for reproducibility."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from app.validation import (
    validate_appointment_datetime,
    validate_full_name,
    validate_text,
    validate_tr_phone,
    validate_value,
)

# Friday 2026-11-20 10:00, Istanbul.
NOW = datetime(2026, 11, 20, 10, 0, tzinfo=ZoneInfo("Europe/Istanbul"))


# ---------------------------------------------------------------- full name


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Ali Veli", "Ali Veli"),
        ("  Ayşe   Gül  Çınar ", "Ayşe Gül Çınar"),
        ("Jean-Luc O'Neil", "Jean-Luc O'Neil"),
    ],
)
def test_valid_names_are_normalized(raw: str, expected: str) -> None:
    assert validate_full_name(raw).value == expected


@pytest.mark.parametrize("raw", ["Ali", "Al", "Ali 123", "x" * 101 + " y", "Ali @Veli"])
def test_invalid_names_are_rejected(raw: str) -> None:
    result = validate_full_name(raw)
    assert not result.ok and result.error


# --------------------------------------------------------------------- phone


@pytest.mark.parametrize(
    "raw",
    [
        "0532 123 45 67",
        "05321234567",
        "5321234567",
        "+90 532 123 45 67",
        "+905321234567",
        "905321234567",
        "0090 532 123 4567",
        "(0532) 123-45-67",
    ],
)
def test_tr_mobile_formats_are_normalized(raw: str) -> None:
    assert validate_tr_phone(raw).value == "0532 123 45 67"


def test_tr_landline_is_accepted() -> None:
    assert validate_tr_phone("0312 555 12 34").value == "0312 555 12 34"


@pytest.mark.parametrize(
    "raw",
    ["0532 123 45", "0532 123 45 678", "0632 123 45 67", "abc", "1234567890", "+1 555 123 4567"],
)
def test_invalid_phones_are_rejected(raw: str) -> None:
    assert not validate_tr_phone(raw).ok


# ------------------------------------------------------------------ datetime


@pytest.mark.parametrize(
    "raw",
    [
        "2026-11-20 14:00",  # later today
        "2026-11-23 09:00",  # Monday opening time
        "2026-11-23 17:30",  # Monday last slot
        "2026-11-21 13:30",  # Saturday last slot
        "2026-11-26T14:00",  # ISO "T" separator is tolerated
    ],
)
def test_valid_appointment_times(raw: str) -> None:
    result = validate_appointment_datetime(raw, NOW)
    assert result.ok, result.error
    assert result.value == raw.replace("T", " ")


@pytest.mark.parametrize(
    ("raw", "error_part"),
    [
        ("2026-11-19 14:00", "past"),
        ("2026-11-20 09:30", "past"),  # earlier today
        ("2026-11-22 12:00", "closed on Sundays"),
        ("2026-11-21 14:00", "between 10:00 and 13:30"),  # Saturday after last slot
        ("2026-11-23 08:30", "between 09:00 and 17:30"),
        ("2026-11-23 18:00", "between 09:00 and 17:30"),
        ("2026-11-23 14:15", "every 30 minutes"),
        ("2027-02-01 10:00", "60 days"),
        ("yarın 14:00", "YYYY-MM-DD HH:MM"),
        ("2026-11-23", "YYYY-MM-DD HH:MM"),
    ],
)
def test_invalid_appointment_times(raw: str, error_part: str) -> None:
    result = validate_appointment_datetime(raw, NOW)
    assert not result.ok
    assert error_part in result.error


# ---------------------------------------------------------------------- text


def test_text_is_whitespace_normalized() -> None:
    assert validate_text("  Sol alt   dişim ağrıyor ").value == "Sol alt dişim ağrıyor"


@pytest.mark.parametrize("raw", ["ab", "x" * 501])
def test_text_length_limits(raw: str) -> None:
    assert not validate_text(raw).ok


# ------------------------------------------------------------------ dispatch


def test_dispatch_by_field_type() -> None:
    assert validate_value("phone", "05321234567", NOW).value == "0532 123 45 67"
    assert not validate_value("appointment_datetime", "2026-11-22 12:00", NOW).ok


def test_unknown_type_falls_back_to_text_check() -> None:
    assert validate_value("something_new", "  ok  ", NOW).value == "ok"
