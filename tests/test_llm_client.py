"""Unit tests for the OpenAI-backed LLM client. No network calls: the SDK is faked."""

from __future__ import annotations

from typing import Any

import pytest
from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseUsage,
)

from app.config import Settings
from app.llm_client import OpenAIClient, ToolSpec, create_llm_client


class FakeResponses:
    def __init__(self, response: Response) -> None:
        self._response = response
        self.last_request: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> Response:
        self.last_request = kwargs
        return self._response


class FakeSDK:
    def __init__(self, response: Response) -> None:
        self.responses = FakeResponses(response)


def make_response(output: list[Any], input_tokens: int = 12, output_tokens: int = 7) -> Response:
    # model_construct skips validation so we only fill the fields we care about.
    usage = ResponseUsage.model_construct(input_tokens=input_tokens, output_tokens=output_tokens)
    return Response.model_construct(model="gpt-test", output=output, usage=usage)


def text_message(text: str) -> ResponseOutputMessage:
    return ResponseOutputMessage.model_construct(
        id="msg_1",
        type="message",
        role="assistant",
        status="completed",
        content=[ResponseOutputText.model_construct(type="output_text", text=text, annotations=[])],
    )


def function_call(name: str, arguments: str, call_id: str = "call_1") -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall.model_construct(
        type="function_call", call_id=call_id, name=name, arguments=arguments
    )


def test_text_response_is_normalized() -> None:
    sdk = FakeSDK(make_response([text_message("Merhaba!")]))
    client = OpenAIClient(model="gpt-test", sdk_client=sdk)

    history = [client.user_message("Selam")]
    result = client.chat(history, system="You are helpful.")

    assert result.text == "Merhaba!"
    assert result.tool_calls == []
    assert result.model == "gpt-test"
    assert (result.input_tokens, result.output_tokens) == (12, 7)
    assert result.latency_ms >= 0
    assert result.history_items[0]["type"] == "message"

    request = sdk.responses.last_request
    assert request["input"] == [{"role": "user", "content": "Selam"}]
    assert request["instructions"] == "You are helpful."
    assert request["store"] is False
    assert "tools" not in request


def test_function_call_is_parsed_and_tools_are_converted() -> None:
    arguments = '{"fields": {"phone": "05321234567"}}'
    sdk = FakeSDK(make_response([function_call("update_fields", arguments)]))
    client = OpenAIClient(model="gpt-test", sdk_client=sdk)
    tool = ToolSpec(
        name="update_fields",
        description="Store extracted form fields.",
        parameters={"type": "object", "properties": {"fields": {"type": "object"}}},
    )

    result = client.chat([client.user_message("0532 123 45 67")], tools=[tool])

    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.id == "call_1"
    assert call.name == "update_fields"
    assert call.arguments == {"fields": {"phone": "05321234567"}}

    sent_tool = sdk.responses.last_request["tools"][0]
    assert sent_tool["type"] == "function"
    assert sent_tool["name"] == "update_fields"
    assert sent_tool["parameters"] == tool.parameters


@pytest.mark.parametrize("raw", ["{not json", "[1, 2]"])
def test_invalid_tool_arguments_become_none(raw: str) -> None:
    sdk = FakeSDK(make_response([function_call("update_fields", raw)]))
    client = OpenAIClient(model="gpt-test", sdk_client=sdk)

    call = client.chat([client.user_message("hi")]).tool_calls[0]

    assert call.arguments is None
    assert call.raw_arguments == raw


def test_tool_result_item_format() -> None:
    client = OpenAIClient(model="gpt-test", sdk_client=FakeSDK(make_response([])))
    assert client.tool_result("call_1", "ok") == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "ok",
    }


def test_create_llm_client_requires_api_key() -> None:
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        create_llm_client(Settings(openai_api_key=None, openai_model="gpt-test"))
