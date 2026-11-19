"""Tools the agent can call, and the code that executes them.

Each tool has two halves:
- a `ToolSpec` (name, description, JSON Schema) that is sent to the LLM, and
- a Python handler that actually does the work against the form state.

Handlers never raise into the agent loop: every failure is returned to the model
as a structured error, so it can correct itself (e.g. re-ask the user).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from app.llm_client import ToolSpec
from app.state import FormState

ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]


def build_update_fields_spec(state: FormState) -> ToolSpec:
    # The schema is generated from the form definition, so a different form
    # (e.g. fetched from Jotform later) automatically gets a matching tool.
    properties = {
        field.name: {"type": "string", "description": field.description} for field in state.fields
    }
    return ToolSpec(
        name="update_fields",
        description=(
            "Save form field values the user has provided. Call this as soon as the user "
            "gives or changes any field value. Only include fields the user actually stated; "
            "never guess or invent values."
        ),
        parameters={
            "type": "object",
            "properties": {
                "fields": {
                    "type": "object",
                    "description": "Field name to value. Include only fields with new values.",
                    "properties": properties,
                    "additionalProperties": False,
                }
            },
            "required": ["fields"],
            "additionalProperties": False,
        },
    )


class ToolExecutor:
    """Holds the tool registry for one conversation and runs tool calls by name."""

    def __init__(self, state: FormState) -> None:
        self.state = state
        self._handlers: dict[str, ToolHandler] = {"update_fields": self._update_fields}
        self.specs: list[ToolSpec] = [build_update_fields_spec(state)]

    def execute(self, name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
        handler = self._handlers.get(name)
        if handler is None:
            return {"ok": False, "error": f"Unknown tool '{name}'."}
        if arguments is None:
            return {"ok": False, "error": "Arguments were not valid JSON. Call the tool again."}
        try:
            return handler(arguments)
        except Exception as error:  # noqa: BLE001 - report any handler bug back to the model
            return {"ok": False, "error": f"Tool failed: {error}"}

    def _update_fields(self, arguments: dict[str, Any]) -> dict[str, Any]:
        fields = arguments.get("fields")
        if not isinstance(fields, dict) or not fields:
            return {"ok": False, "error": "'fields' must be a non-empty object."}
        result = self.state.update(fields)
        return {
            "ok": not result.errors,
            "updated": result.updated,
            "errors": result.errors,
            # Returning the current state tells the model what to ask for next.
            "current_values": dict(self.state.values),
            "missing_fields": self.state.missing_fields(),
        }


def to_tool_output(result: dict[str, Any]) -> str:
    """Serialize a tool result for the LLM (keeps Turkish characters readable)."""
    return json.dumps(result, ensure_ascii=False)
