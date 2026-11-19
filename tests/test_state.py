from app.state import new_dental_form


def test_update_stores_known_fields_and_strips_whitespace() -> None:
    state = new_dental_form()

    result = state.update({"full_name": "  Ali Veli ", "phone": "05321234567"})

    assert result.updated == ["full_name", "phone"]
    assert result.errors == {}
    assert state.values == {"full_name": "Ali Veli", "phone": "05321234567"}


def test_update_rejects_unknown_and_empty_fields() -> None:
    state = new_dental_form()

    result = state.update({"email": "a@b.com", "complaint": "   ", "full_name": None})

    assert result.updated == []
    assert set(result.errors) == {"email", "complaint", "full_name"}
    assert state.values == {}


def test_later_update_overwrites_value() -> None:
    state = new_dental_form()
    state.update({"appointment_datetime": "2026-11-21 14:00"})

    state.update({"appointment_datetime": "2026-11-26 14:00"})

    assert state.values["appointment_datetime"] == "2026-11-26 14:00"


def test_missing_fields_and_completion() -> None:
    state = new_dental_form()
    assert state.missing_fields() == ["full_name", "phone", "appointment_datetime", "complaint"]
    assert not state.is_complete()

    state.update(
        {
            "full_name": "Ali Veli",
            "phone": "05321234567",
            "appointment_datetime": "2026-11-21 14:00",
            "complaint": "Toothache",
        }
    )

    assert state.missing_fields() == []
    assert state.is_complete()


def test_version_only_changes_when_a_value_changes() -> None:
    state = new_dental_form()
    state.update({"full_name": "Ali Veli"})
    assert state.version == 1

    state.update({"full_name": "Ali Veli"})  # same value again
    assert state.version == 1

    state.update({"full_name": "Ali Can Veli"})
    assert state.version == 2
