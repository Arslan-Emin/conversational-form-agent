"""Jotform API integration: client, form schema loading, and submitters.

Verified against Jotform's official Python client (github.com/jotform/jotform-api-python):
- endpoints live under `/v1`, the API key goes in the `APIKEY` header,
- responses wrap the payload in a `content` key,
- submissions are POSTed form-urlencoded as `submission[<qid>]=value`, and
  compound fields use `submission[<qid>][<subfield>]=value`.

Sub-field names for compound question types are not documented there. They were
verified with live test submissions (read back via GET /submission/{id}, then
deleted) on a form with Full Name, Phone (input mask) and Date/Time (24h) fields:
- control_fullname -> [first], [last]
- control_phone (masked) -> [full], formatted with the question's input mask
- control_datetime -> [year], [month], [day], [hour], [min]; Jotform parses these
  into its own datetime value.
The phone-without-mask and AM/PM variants follow the same conventions but were not
tested live.

Run `python -m app.jotform questions` to print the configured form's questions.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Protocol

import httpx

from app.config import get_settings
from app.state import FieldSpec, FormState

API_VERSION_PREFIX = "/v1"


class JotformError(RuntimeError):
    """Raised for any failed Jotform API call. Never contains the API key."""


class JotformClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.jotform.com",
        *,
        http: httpx.Client | None = None,
        timeout: float = 15.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        # `http` lets tests inject an httpx.Client with a MockTransport.
        self._http = http or httpx.Client(timeout=timeout)

    def _request(self, method: str, path: str, data: dict[str, str] | None = None) -> Any:
        url = f"{self._base_url}{API_VERSION_PREFIX}{path}"
        try:
            response = self._http.request(method, url, headers={"APIKEY": self._api_key}, data=data)
        except httpx.HTTPError as error:
            raise JotformError(f"Could not reach Jotform: {error.__class__.__name__}") from error

        try:
            payload = response.json()
        except ValueError as error:
            raise JotformError(
                f"Jotform returned non-JSON (HTTP {response.status_code})."
            ) from error

        response_code = payload.get("responseCode", response.status_code)
        if response.status_code != 200 or response_code != 200:
            message = payload.get("message") or "unknown error"
            raise JotformError(f"Jotform API error (HTTP {response.status_code}): {message}")
        return payload.get("content")

    def get_questions(self, form_id: str) -> dict[str, dict[str, Any]]:
        content = self._request("GET", f"/form/{form_id}/questions")
        if not isinstance(content, dict):
            raise JotformError("Unexpected questions format from Jotform.")
        return content

    def create_submission(self, form_id: str, data: dict[str, str]) -> dict[str, Any]:
        content = self._request("POST", f"/form/{form_id}/submissions", data=data)
        return content if isinstance(content, dict) else {}


# --------------------------------------------------------------------------- #
# Schema: Jotform questions -> our FieldSpec list
# --------------------------------------------------------------------------- #

# Jotform question type -> our validator type (see app/validation.py).
FIELD_TYPE_BY_JOTFORM_TYPE = {
    "control_fullname": "full_name",
    "control_phone": "phone",
    "control_datetime": "appointment_datetime",
    "control_textbox": "text",
    "control_textarea": "text",
}

# Layout elements and buttons: not something the user answers.
NON_INPUT_TYPES = {
    "control_head",
    "control_button",
    "control_text",
    "control_divider",
    "control_pagebreak",
    "control_image",
    "control_collapse",
    "control_separator",
}

_TYPE_HINTS = {
    "full_name": "First and last name.",
    "phone": "Turkish phone number.",
    "appointment_datetime": "Exact local date and time, format YYYY-MM-DD HH:MM.",
}


def fields_from_questions(questions: dict[str, dict[str, Any]]) -> list[FieldSpec]:
    """Convert Jotform's `/questions` response into field specs, in form order."""
    ordered = sorted(questions.values(), key=lambda q: int(q.get("order", 0) or 0))
    fields: list[FieldSpec] = []
    for question in ordered:
        jotform_type = str(question.get("type", ""))
        if jotform_type in NON_INPUT_TYPES or not jotform_type.startswith("control_"):
            continue
        field_type = FIELD_TYPE_BY_JOTFORM_TYPE.get(jotform_type, "text")
        label = str(question.get("text") or question.get("name") or f"Question {question['qid']}")
        hint = _TYPE_HINTS.get(field_type, "")
        fields.append(
            FieldSpec(
                name=str(question.get("name") or f"q{question['qid']}"),
                label=label,
                description=f"{label}. {hint}".strip(),
                type=field_type,
                required=str(question.get("required", "No")).lower() == "yes",
                qid=str(question["qid"]),
                jotform_type=jotform_type,
                jotform_props=question,
            )
        )
    return fields


def load_form_from_jotform(client: JotformClient, form_id: str) -> FormState:
    fields = fields_from_questions(client.get_questions(form_id))
    if not fields:
        raise JotformError(f"Form {form_id} has no input questions.")
    return FormState(fields=fields)


# --------------------------------------------------------------------------- #
# Payload: our validated values -> Jotform submission fields
# --------------------------------------------------------------------------- #


def apply_input_mask(digits: str, mask: object) -> str:
    """Fill a Jotform input mask such as "(###) ###-####" with digits.

    Falls back to the plain digits if the mask does not have one slot per digit,
    so we never send a half-filled mask.
    """
    if not isinstance(mask, str) or mask.count("#") != len(digits):
        return digits
    remaining = iter(digits)
    return "".join(next(remaining) if char == "#" else char for char in mask)


def build_answer_parts(field: FieldSpec, value: str) -> dict[str, str] | str:
    """Split a normalized value into Jotform sub-fields where the question type needs it.

    Returns a plain string for simple questions, or {subfield: value} for compound ones.
    Input values are already normalized by app/validation.py.
    """
    if field.jotform_type == "control_fullname":
        first, _, last = value.rpartition(" ")
        return {"first": first, "last": last}

    if field.jotform_type == "control_phone":
        # Normalized as "0532 123 45 67". With an input mask Jotform stores one "full"
        # value; without one it has separate area code and number boxes.
        digits = value.replace(" ", "")[1:]  # 10 digits without the leading 0
        if field.jotform_props.get("inputMask") == "enable":
            return {"full": apply_input_mask(digits, field.jotform_props.get("inputMaskValue"))}
        return {"area": digits[:3], "phone": digits[3:]}

    if field.jotform_type == "control_datetime":
        # Normalized as "YYYY-MM-DD HH:MM".
        date_part, time_part = value.split(" ")
        year, month, day = date_part.split("-")
        hour, minute = time_part.split(":")
        parts = {"year": year, "month": month, "day": day, "hour": hour, "min": minute}
        if str(field.jotform_props.get("timeFormat", "")).upper().startswith("AM"):
            hour_int = int(hour)
            parts["hour"] = f"{(hour_int % 12) or 12:02d}"
            parts["ampm"] = "AM" if hour_int < 12 else "PM"
        return parts

    return value


def build_submission_payload(state: FormState) -> dict[str, str]:
    """Form-urlencoded body for POST /form/{id}/submissions."""
    payload: dict[str, str] = {}
    for field in state.fields:
        value = state.values.get(field.name)
        if value is None:
            continue
        key = field.qid or field.name  # local (non-Jotform) forms fall back to the field name
        parts = build_answer_parts(field, value)
        if isinstance(parts, dict):
            for sub_key, sub_value in parts.items():
                payload[f"submission[{key}][{sub_key}]"] = sub_value
        else:
            payload[f"submission[{key}]"] = parts
    return payload


# --------------------------------------------------------------------------- #
# Submitters
# --------------------------------------------------------------------------- #


class FormSubmitter(Protocol):
    def submit(self, state: FormState) -> dict[str, Any]: ...


class DryRunSubmitter:
    """Builds the exact payload but sends nothing. Used in tests, evals and --dry-run."""

    def __init__(self) -> None:
        self.submissions: list[dict[str, str]] = []

    def submit(self, state: FormState) -> dict[str, Any]:
        payload = build_submission_payload(state)
        self.submissions.append(payload)
        return {"dry_run": True, "submission_id": None, "payload": payload}


class JotformSubmitter:
    def __init__(self, client: JotformClient, form_id: str) -> None:
        self._client = client
        self._form_id = form_id

    def submit(self, state: FormState) -> dict[str, Any]:
        content = self._client.create_submission(self._form_id, build_submission_payload(state))
        submission_id = content.get("submissionID")
        return {"dry_run": False, "submission_id": str(submission_id) if submission_id else None}


# --------------------------------------------------------------------------- #
# Small CLI to inspect the configured form
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if args != ["questions"]:
        print("Usage: python -m app.jotform questions", file=sys.stderr)
        return 2
    settings = get_settings()
    if not settings.jotform_configured:
        print("Set JOTFORM_API_KEY and JOTFORM_FORM_ID in .env first.", file=sys.stderr)
        return 1
    client = JotformClient(settings.jotform_api_key, settings.jotform_base_url)
    try:
        questions = client.get_questions(settings.jotform_form_id)
    except JotformError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(questions, ensure_ascii=False, indent=2))
    print("\nMapped fields:")
    for field in fields_from_questions(questions):
        print(f"  qid={field.qid} {field.jotform_type} -> name={field.name!r} type={field.type}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
