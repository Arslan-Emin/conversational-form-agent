"""Shared setup for the CLI and the Streamlit app: which form, and where submissions go."""

from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings, get_settings
from app.jotform import (
    DryRunSubmitter,
    FormSubmitter,
    JotformClient,
    JotformError,
    JotformSubmitter,
    fields_from_questions,
)
from app.state import DENTAL_APPOINTMENT_FIELDS, FieldSpec, FormState


@dataclass(frozen=True)
class FormSetup:
    """The form definition, loaded once and shared by every conversation."""

    fields: list[FieldSpec]
    description: str
    jotform_client: JotformClient | None = None
    jotform_form_id: str | None = None

    @property
    def can_submit_live(self) -> bool:
        return self.jotform_client is not None

    def new_state(self) -> FormState:
        # Copies, so conversations never share mutable field objects.
        return FormState(fields=[field.model_copy(deep=True) for field in self.fields])

    def create_submitter(self, live: bool) -> FormSubmitter:
        if live and self.jotform_client is not None and self.jotform_form_id:
            return JotformSubmitter(self.jotform_client, self.jotform_form_id)
        return DryRunSubmitter()


def load_form_setup(settings: Settings | None = None) -> FormSetup:
    """Use the Jotform form if configured, otherwise the built-in example form.

    Raises JotformError if Jotform is configured but unreachable or misconfigured:
    silently falling back would hide a broken setup.
    """
    settings = settings or get_settings()
    if not settings.jotform_configured:
        return FormSetup(fields=list(DENTAL_APPOINTMENT_FIELDS), description="example form")

    client = JotformClient(settings.jotform_api_key, settings.jotform_base_url)
    fields = fields_from_questions(client.get_questions(settings.jotform_form_id))
    if not fields:
        raise JotformError(f"Form {settings.jotform_form_id} has no input questions.")
    return FormSetup(
        fields=fields,
        description=f"Jotform form {settings.jotform_form_id}",
        jotform_client=client,
        jotform_form_id=settings.jotform_form_id,
    )
