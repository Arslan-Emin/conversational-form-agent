"""Jotform client and mapping tests. All HTTP is mocked with httpx.MockTransport."""

from __future__ import annotations

import json
from urllib.parse import parse_qsl

import httpx
import pytest

from app.jotform import (
    DryRunSubmitter,
    JotformClient,
    JotformError,
    JotformSubmitter,
    apply_input_mask,
    build_submission_payload,
    fields_from_questions,
    load_form_from_jotform,
)
from app.state import FormState, new_dental_form

QUESTIONS = {
    "1": {"qid": "1", "type": "control_head", "name": "heading", "text": "Booking", "order": "1"},
    "3": {
        "qid": "3",
        "type": "control_fullname",
        "name": "fullName",
        "text": "Full Name",
        "order": "2",
        "required": "Yes",
    },
    "4": {
        "qid": "4",
        "type": "control_phone",
        "name": "phoneNumber",
        "text": "Phone Number",
        "order": "3",
        "required": "Yes",
        "inputMask": "enable",
        "inputMaskValue": "(###) ###-####",
    },
    "5": {
        "qid": "5",
        "type": "control_datetime",
        "name": "appointment",
        "text": "Appointment",
        "order": "4",
        "required": "Yes",
        "timeFormat": "24 Hour",
    },
    "6": {
        "qid": "6",
        "type": "control_textarea",
        "name": "complaint",
        "text": "Complaint",
        "order": "5",
        "required": "No",
    },
    "2": {"qid": "2", "type": "control_button", "name": "submit", "text": "Submit", "order": "6"},
}

VALUES = {
    "fullName": "Ayşe Gül Çınar",
    "phoneNumber": "0532 123 45 67",
    "appointment": "2026-11-26 14:30",
    "complaint": "Sol alt dişte ağrı",
}


def make_client(handler) -> JotformClient:
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return JotformClient("secret-key", "https://api.jotform.com/", http=http)


def ok(content) -> httpx.Response:
    return httpx.Response(200, json={"responseCode": 200, "message": "success", "content": content})


# ------------------------------------------------------------------ client


def test_get_questions_uses_v1_path_and_apikey_header() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return ok(QUESTIONS)

    questions = make_client(handler).get_questions("123")

    assert questions == QUESTIONS
    request = seen[0]
    assert request.method == "GET"
    assert str(request.url) == "https://api.jotform.com/v1/form/123/questions"
    assert request.headers["APIKEY"] == "secret-key"


def test_create_submission_posts_form_urlencoded_body() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return ok({"submissionID": "999", "URL": "https://api.jotform.com/submission/999"})

    content = make_client(handler).create_submission("123", {"submission[3][first]": "Ali"})

    assert content["submissionID"] == "999"
    request = seen[0]
    assert request.method == "POST"
    assert request.headers["content-type"].startswith("application/x-www-form-urlencoded")
    assert dict(parse_qsl(request.content.decode())) == {"submission[3][first]": "Ali"}


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(401, json={"responseCode": 401, "message": "You're not authorized"}),
        httpx.Response(200, json={"responseCode": 400, "message": "Bad request"}),
        httpx.Response(502, text="<html>Bad gateway</html>"),
    ],
)
def test_api_errors_raise_without_leaking_key(response: httpx.Response) -> None:
    client = make_client(lambda request: response)

    with pytest.raises(JotformError) as error:
        client.get_questions("123")

    assert "secret-key" not in str(error.value)


def test_network_error_is_wrapped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    with pytest.raises(JotformError, match="Could not reach Jotform"):
        make_client(handler).get_questions("123")


# ------------------------------------------------------------------ schema


def test_questions_are_mapped_to_fields_in_form_order() -> None:
    fields = fields_from_questions(QUESTIONS)

    assert [f.name for f in fields] == ["fullName", "phoneNumber", "appointment", "complaint"]
    assert [f.type for f in fields] == ["full_name", "phone", "appointment_datetime", "text"]
    assert [f.qid for f in fields] == ["3", "4", "5", "6"]
    assert [f.required for f in fields] == [True, True, True, False]
    assert "YYYY-MM-DD HH:MM" in fields[2].description


def test_load_form_rejects_form_without_inputs() -> None:
    client = make_client(lambda request: ok({"1": QUESTIONS["1"]}))

    with pytest.raises(JotformError, match="no input questions"):
        load_form_from_jotform(client, "123")


# ------------------------------------------------------------------ payload


def jotform_state() -> FormState:
    state = FormState(fields=fields_from_questions(QUESTIONS))
    state.values.update(VALUES)
    return state


def test_payload_splits_compound_fields() -> None:
    assert build_submission_payload(jotform_state()) == {
        "submission[3][first]": "Ayşe Gül",
        "submission[3][last]": "Çınar",
        "submission[4][full]": "(532) 123-4567",
        "submission[5][year]": "2026",
        "submission[5][month]": "11",
        "submission[5][day]": "26",
        "submission[5][hour]": "14",
        "submission[5][min]": "30",
        "submission[6]": "Sol alt dişte ağrı",
    }


def test_phone_without_mask_uses_area_and_number() -> None:
    questions = json.loads(json.dumps(QUESTIONS))
    del questions["4"]["inputMask"]
    state = FormState(fields=fields_from_questions(questions))
    state.values.update(VALUES)

    payload = build_submission_payload(state)

    assert payload["submission[4][area]"] == "532"
    assert payload["submission[4][phone]"] == "1234567"


def test_datetime_with_am_pm_format() -> None:
    questions = json.loads(json.dumps(QUESTIONS))
    questions["5"]["timeFormat"] = "AM/PM"
    state = FormState(fields=fields_from_questions(questions))
    state.values.update(VALUES)

    payload = build_submission_payload(state)

    assert payload["submission[5][hour]"] == "02"
    assert payload["submission[5][ampm]"] == "PM"


def test_local_form_payload_falls_back_to_field_names() -> None:
    state = new_dental_form()
    state.values["complaint"] = "Kontrol"

    assert build_submission_payload(state) == {"submission[complaint]": "Kontrol"}


# ------------------------------------------------------------------ submitters


def test_dry_run_submitter_sends_nothing() -> None:
    submitter = DryRunSubmitter()

    receipt = submitter.submit(jotform_state())

    assert receipt["dry_run"] is True
    assert receipt["payload"]["submission[6]"] == "Sol alt dişte ağrı"
    assert len(submitter.submissions) == 1


def test_jotform_submitter_returns_submission_id() -> None:
    client = make_client(lambda request: ok({"submissionID": 123456789}))

    receipt = JotformSubmitter(client, "123").submit(jotform_state())

    assert receipt == {"dry_run": False, "submission_id": "123456789"}


@pytest.mark.parametrize(
    ("mask", "expected"),
    [
        ("(###) ###-####", "(532) 123-4567"),
        ("### ### ## ##", "532 123 45 67"),
        ("+90 (###) ### ####", "+90 (532) 123 4567"),
        ("(###) ###-###", "5321234567"),  # wrong slot count: fall back to digits
        (None, "5321234567"),
    ],
)
def test_apply_input_mask(mask, expected) -> None:
    assert apply_input_mask("5321234567", mask) == expected
