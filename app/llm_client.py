"""Thin, provider-agnostic wrapper around LLM APIs.

Every LLM call in the project goes through this module. That gives us one place
to measure tokens and latency today, and to add cost tracking and a trace hook
later (step 8), without touching the agent code.

The agent only depends on the `LLMClient` protocol and the small dataclasses
below. Conversation history items are provider-specific dicts, but the agent
never builds them by hand: it asks the client for them (`user_message`,
`tool_result`, `LLMResponse.history_items`). Swapping OpenAI for another
provider therefore means writing one new class, not rewriting the agent.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from openai import OpenAI

from app.config import Settings, get_settings

HistoryItem = dict[str, Any]


@dataclass(frozen=True)
class ToolSpec:
    """A tool the model may call. `parameters` is a JSON Schema object."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """A tool call requested by the model.

    `arguments` is None when the model produced invalid JSON; the agent can then
    report the error back to the model instead of crashing.
    """

    id: str
    name: str
    arguments: dict[str, Any] | None
    raw_arguments: str


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    # Provider-format items to append to the conversation history as-is.
    history_items: list[HistoryItem] = field(default_factory=list)


class LLMClient(Protocol):
    model: str

    def user_message(self, text: str) -> HistoryItem: ...

    def tool_result(self, call_id: str, output: str) -> HistoryItem: ...

    def chat(
        self,
        history: list[HistoryItem],
        *,
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse: ...


class OpenAIClient:
    """`LLMClient` implementation on top of the OpenAI Responses API.

    We call the API statelessly (`store=False`) and send the full history on
    every request: the conversation state lives in our code, not on OpenAI's
    servers. That keeps runs reproducible for evals and avoids storing user data.
    """

    def __init__(self, model: str, api_key: str | None = None, sdk_client: Any = None) -> None:
        self.model = model
        # `sdk_client` lets tests inject a fake instead of hitting the network.
        self._client = sdk_client if sdk_client is not None else OpenAI(api_key=api_key)

    def user_message(self, text: str) -> HistoryItem:
        return {"role": "user", "content": text}

    def tool_result(self, call_id: str, output: str) -> HistoryItem:
        return {"type": "function_call_output", "call_id": call_id, "output": output}

    def chat(
        self,
        history: list[HistoryItem],
        *,
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        request: dict[str, Any] = {
            "model": self.model,
            "input": history,
            "store": False,
            # Reasoning models return reasoning items; with store=False they must be
            # sent back in encrypted form so the model keeps its context across turns.
            "include": ["reasoning.encrypted_content"],
        }
        if system:
            request["instructions"] = system
        if tools:
            request["tools"] = [_to_openai_tool(tool) for tool in tools]

        start = time.perf_counter()
        response = self._client.responses.create(**request)
        latency_ms = (time.perf_counter() - start) * 1000

        tool_calls = [
            ToolCall(
                id=item.call_id,
                name=item.name,
                arguments=_parse_arguments(item.arguments),
                raw_arguments=item.arguments,
            )
            for item in response.output
            if item.type == "function_call"
        ]
        usage = response.usage
        return LLMResponse(
            text=response.output_text,
            tool_calls=tool_calls,
            model=response.model,
            input_tokens=usage.input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
            latency_ms=latency_ms,
            history_items=[
                item.model_dump(mode="json", exclude_none=True) for item in response.output
            ],
        )


def _to_openai_tool(tool: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
        "strict": False,
    }


def _parse_arguments(raw: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def create_llm_client(settings: Settings | None = None) -> LLMClient:
    """Build the configured LLM client. Fails fast with a clear message if misconfigured."""
    settings = settings or get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in.")
    return OpenAIClient(model=settings.openai_model, api_key=settings.openai_api_key)
