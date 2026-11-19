"""Form schema and form state.

The state lives in code, not in the LLM's memory. The model can only change it
through tools, and every other part of the system (validation, submission, UI,
evals) reads it from here. This is what makes the agent's behaviour checkable.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class FieldSpec(BaseModel):
    """One form field, described so that both code and the LLM understand it."""

    name: str
    label: str
    description: str
    # Selects the validator in app/validation.py (e.g. "phone", "text").
    type: str = "text"
    required: bool = True


class FieldUpdateResult(BaseModel):
    updated: list[str] = Field(default_factory=list)
    errors: dict[str, str] = Field(default_factory=dict)


class FormState(BaseModel):
    fields: list[FieldSpec]
    values: dict[str, str] = Field(default_factory=dict)

    @property
    def field_names(self) -> list[str]:
        return [field.name for field in self.fields]

    def get_field(self, name: str) -> FieldSpec | None:
        return next((field for field in self.fields if field.name == name), None)

    def update(self, updates: dict[str, object]) -> FieldUpdateResult:
        """Apply field updates. Unknown fields and empty values are rejected, not stored."""
        result = FieldUpdateResult()
        known = set(self.field_names)
        for name, raw_value in updates.items():
            if name not in known:
                result.errors[name] = f"Unknown field. Valid fields: {', '.join(self.field_names)}"
                continue
            value = str(raw_value).strip() if raw_value is not None else ""
            if not value:
                result.errors[name] = "Empty value. Ask the user for this field instead."
                continue
            self.values[name] = value
            result.updated.append(name)
        return result

    def missing_fields(self) -> list[str]:
        return [f.name for f in self.fields if f.required and f.name not in self.values]

    def is_complete(self) -> bool:
        return not self.missing_fields()


# Example form used until the schema is fetched from the Jotform API (step 4).
DENTAL_APPOINTMENT_FIELDS = [
    FieldSpec(
        name="full_name",
        type="full_name",
        label="Full name",
        description="Patient's first and last name.",
    ),
    FieldSpec(
        name="phone",
        type="phone",
        label="Phone number",
        description="Patient's Turkish phone number.",
    ),
    FieldSpec(
        name="appointment_datetime",
        type="appointment_datetime",
        label="Appointment date and time",
        description="Requested appointment as an exact local date and time.",
    ),
    FieldSpec(
        name="complaint",
        type="text",
        label="Complaint",
        description="Short description of the dental problem or reason for the visit.",
    ),
]


def new_dental_form() -> FormState:
    return FormState(fields=[field.model_copy() for field in DENTAL_APPOINTMENT_FIELDS])
