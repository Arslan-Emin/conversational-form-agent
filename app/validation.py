"""Deterministic field validation and normalization.

The LLM extracts values from the conversation, but it can misread, hallucinate or
be talked into things. Every value therefore passes through plain Python checks
before it is stored. A check either returns a normalized value or a clear error
message that the model relays to the user.

Validators are picked by field *type* (not by field name), so forms fetched from
Jotform later can reuse them.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, time, timedelta

# --------------------------------------------------------------------------- #
# Clinic rules. Single source of truth: the system prompt is generated from
# these, so the model and the validator can never disagree.
# --------------------------------------------------------------------------- #

# weekday() -> (opening, closing); missing day = closed. Monday is 0.
OPENING_HOURS: dict[int, tuple[time, time]] = {
    0: (time(9, 0), time(18, 0)),
    1: (time(9, 0), time(18, 0)),
    2: (time(9, 0), time(18, 0)),
    3: (time(9, 0), time(18, 0)),
    4: (time(9, 0), time(18, 0)),
    5: (time(10, 0), time(14, 0)),
}
SLOT_MINUTES = 30
MAX_DAYS_AHEAD = 60
DATETIME_FORMAT = "%Y-%m-%d %H:%M"

WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def describe_opening_hours() -> str:
    """Human-readable opening hours, e.g. for the system prompt."""
    lines = []
    for day, name in enumerate(WEEKDAY_NAMES):
        hours = OPENING_HOURS.get(day)
        if hours:
            lines.append(f"{name}: {hours[0]:%H:%M}-{hours[1]:%H:%M}")
        else:
            lines.append(f"{name}: closed")
    return "; ".join(lines)


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    value: str | None = None
    error: str | None = None

    @classmethod
    def valid(cls, value: str) -> ValidationResult:
        return cls(ok=True, value=value)

    @classmethod
    def invalid(cls, error: str) -> ValidationResult:
        return cls(ok=False, error=error)


# --------------------------------------------------------------------------- #
# Validators
# --------------------------------------------------------------------------- #

_NAME_EXTRA_CHARS = set(" -'.")


def validate_full_name(value: str) -> ValidationResult:
    name = " ".join(value.split())
    if not 3 <= len(name) <= 100:
        return ValidationResult.invalid("Name must be between 3 and 100 characters.")
    if not all(ch.isalpha() or ch in _NAME_EXTRA_CHARS for ch in name):
        return ValidationResult.invalid("Name may only contain letters, spaces, '-' and '.'.")
    if len(name.split()) < 2:
        return ValidationResult.invalid("Please provide both first name and last name.")
    return ValidationResult.valid(name)


_PHONE_SEPARATORS = re.compile(r"[\s\-().]")


def validate_tr_phone(value: str) -> ValidationResult:
    """Accept common Turkish formats (+90 / 90 / 0 prefix, spaces, dashes).

    Normalized form: `0532 123 45 67`.
    """
    digits = _PHONE_SEPARATORS.sub("", value)
    if digits.startswith("+90"):
        digits = digits[3:]
    elif digits.startswith("0090"):
        digits = digits[4:]
    elif digits.startswith("90") and len(digits) == 12:
        digits = digits[2:]
    elif digits.startswith("0"):
        digits = digits[1:]

    if not digits.isdigit() or len(digits) != 10:
        return ValidationResult.invalid(
            "Phone number must be a Turkish number with 10 digits after the leading 0, "
            "e.g. 0532 123 45 67."
        )
    # Turkish area codes start with 2-4 (landline) or 5 (mobile).
    if digits[0] not in "2345":
        return ValidationResult.invalid("This does not look like a valid Turkish phone number.")
    return ValidationResult.valid(f"0{digits[:3]} {digits[3:6]} {digits[6:8]} {digits[8:]}")


def validate_appointment_datetime(value: str, now: datetime) -> ValidationResult:
    """`value` must be `YYYY-MM-DD HH:MM` local time, in the future, within opening hours."""
    try:
        requested = datetime.strptime(value.strip().replace("T", " "), DATETIME_FORMAT)
    except ValueError:
        return ValidationResult.invalid(
            "Date/time must be in 'YYYY-MM-DD HH:MM' format. "
            "If the user did not give an exact time, ask for it."
        )
    requested = requested.replace(tzinfo=now.tzinfo)

    if requested <= now:
        return ValidationResult.invalid("This date/time is in the past.")
    if requested > now + timedelta(days=MAX_DAYS_AHEAD):
        return ValidationResult.invalid(
            f"Appointments can be booked at most {MAX_DAYS_AHEAD} days in advance."
        )

    hours = OPENING_HOURS.get(requested.weekday())
    day_name = WEEKDAY_NAMES[requested.weekday()]
    if hours is None:
        return ValidationResult.invalid(
            f"The clinic is closed on {day_name}s. Opening hours: {describe_opening_hours()}."
        )
    opening, closing = hours
    last_start = (
        datetime.combine(requested.date(), closing) - timedelta(minutes=SLOT_MINUTES)
    ).time()
    if not opening <= requested.time() <= last_start:
        return ValidationResult.invalid(
            f"On {day_name}s appointments can start between {opening:%H:%M} and {last_start:%H:%M}."
        )
    if requested.minute % SLOT_MINUTES != 0:
        return ValidationResult.invalid(
            f"Appointments start on the hour or half hour (every {SLOT_MINUTES} minutes)."
        )
    return ValidationResult.valid(requested.strftime(DATETIME_FORMAT))


def validate_text(value: str, min_length: int = 3, max_length: int = 500) -> ValidationResult:
    text = " ".join(value.split())
    if len(text) < min_length:
        return ValidationResult.invalid(
            f"Please give a bit more detail (at least {min_length} characters)."
        )
    if len(text) > max_length:
        return ValidationResult.invalid(f"Text is too long (max {max_length} characters).")
    return ValidationResult.valid(text)


# --------------------------------------------------------------------------- #
# Dispatch by field type
# --------------------------------------------------------------------------- #

Validator = Callable[[str, datetime], ValidationResult]

VALIDATORS: dict[str, Validator] = {
    "full_name": lambda value, _now: validate_full_name(value),
    "phone": lambda value, _now: validate_tr_phone(value),
    "appointment_datetime": validate_appointment_datetime,
    "text": lambda value, _now: validate_text(value),
}


def validate_value(field_type: str, value: str, now: datetime) -> ValidationResult:
    """Validate and normalize `value` for a field of `field_type`.

    `now` is passed in (not read from the system clock) so tests and evals are
    deterministic.
    """
    validator = VALIDATORS.get(field_type)
    if validator is None:
        # Unknown types still get basic text checks rather than none at all.
        return validate_text(value, min_length=1)
    return validator(value, now)
