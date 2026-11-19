"""Agent loop tests with a scripted fake LLM: no network, fully deterministic."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from app.agent import CLINIC_TIMEZONE, ITERATION_LIMIT_REPLY, FormAgent
from app.llm_client import HistoryItem, LLMResponse, ToolCall, ToolSpec

FIXED_NOW = datetime(2026, 11, 20, 10, 0, tzinfo=CLINIC_TIMEZONE)


def text_reply(text: str) -> LLMResponse:
    return LLMResponse(
        text=text,
        tool_calls=[],
        model="fake",
        input_tokens=10,
        output_tokens=5,
        latency_ms=1.0,
        history_items=[{"role": "assistant", "content": text}],
    )


def tool_reply(name: str, arguments: dict[str, Any] | None, call_id: str = "c1") -> LLMResponse:
    call = ToolCall(id=call_id, name=name, arguments=arguments, raw_arguments="...")
    return LLMResponse(
        text="",
        tool_calls=[call],
        model="fake",
        input_tokens=10,
        output_tokens=5,
        latency_ms=1.0,
        history_items=[{"role": "assistant", "tool_calls": [call_id]}],
    )


class ScriptedLLM:
    """Returns pre-written responses in order and records what it was sent."""

    model = "fake"

    def __init__(self, responses: list[LLMResponse | Exception]) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def user_message(self, text: str) -> HistoryItem:
        return {"role": "user", "content": text}

    def assistant_message(self, text: str) -> HistoryItem:
        return {"role": "assistant", "content": text}

    def tool_result(self, call_id: str, output: str) -> HistoryItem:
        return {"role": "tool", "tool_call_id": call_id, "content": output}

    def chat(
        self,
        history: list[HistoryItem],
        *,
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        self.requests.append({"history": list(history), "system": system, "tools": tools})
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def make_agent(responses: list[LLMResponse | Exception], **kwargs: Any) -> FormAgent:
    return FormAgent(ScriptedLLM(responses), clock=lambda: FIXED_NOW, **kwargs)


def test_plain_reply_needs_one_llm_call() -> None:
    agent = make_agent([text_reply("Merhaba! Adınız nedir?")])

    turn = agent.handle_user_message("Merhaba")

    assert turn.reply == "Merhaba! Adınız nedir?"
    assert turn.llm_calls == 1
    assert turn.tool_calls == []
    assert agent.state.values == {}


def test_tool_call_updates_state_then_model_answers() -> None:
    agent = make_agent(
        [
            tool_reply("update_fields", {"fields": {"full_name": "Ali Veli"}}),
            text_reply("Teşekkürler Ali Bey, telefon numaranız?"),
        ]
    )

    turn = agent.handle_user_message("Ben Ali Veli")

    assert agent.state.values == {"full_name": "Ali Veli"}
    assert turn.llm_calls == 2
    assert turn.tool_calls[0].result["ok"] is True
    assert turn.tool_calls[0].result["missing_fields"] == [
        "phone",
        "appointment_datetime",
        "complaint",
    ]
    assert (turn.input_tokens, turn.output_tokens) == (20, 10)
    # History: user, assistant tool call, tool result, final assistant text.
    roles = [item.get("role") for item in agent.history]
    assert roles == ["user", "assistant", "tool", "assistant"]


def test_tool_error_is_reported_back_to_model() -> None:
    agent = make_agent(
        [
            tool_reply("update_fields", None),  # invalid JSON from the model
            text_reply("Pardon, tekrar alayım."),
        ]
    )

    turn = agent.handle_user_message("Ali Veli")

    assert turn.tool_calls[0].result["ok"] is False
    assert "JSON" in turn.tool_calls[0].result["error"]
    tool_message = agent.history[2]
    assert tool_message["role"] == "tool" and '"ok": false' in tool_message["content"]


def test_unknown_tool_does_not_crash() -> None:
    agent = make_agent([tool_reply("delete_everything", {}), text_reply("Tamam.")])

    turn = agent.handle_user_message("x")

    assert turn.tool_calls[0].result == {"ok": False, "error": "Unknown tool 'delete_everything'."}
    assert turn.reply == "Tamam."


def test_iteration_limit_stops_endless_tool_calls() -> None:
    looping = [tool_reply("update_fields", {"fields": {"complaint": "ağrı"}}) for _ in range(3)]
    agent = make_agent(looping, max_iterations=3)

    turn = agent.handle_user_message("Dişim ağrıyor")

    assert turn.hit_iteration_limit is True
    assert turn.llm_calls == 3
    assert turn.reply == ITERATION_LIMIT_REPLY
    assert agent.history[-1] == {"role": "assistant", "content": ITERATION_LIMIT_REPLY}


def test_llm_error_rolls_back_history() -> None:
    agent = make_agent(
        [
            text_reply("Merhaba!"),
            tool_reply("update_fields", {"fields": {"full_name": "Ali"}}),
            RuntimeError("network down"),
        ]
    )
    agent.handle_user_message("Selam")
    history_before = list(agent.history)

    with pytest.raises(RuntimeError):
        agent.handle_user_message("Ben Ali")

    assert agent.history == history_before


def test_system_prompt_contains_pinned_date_and_fields() -> None:
    llm = ScriptedLLM([text_reply("ok")])
    agent = FormAgent(llm, clock=lambda: FIXED_NOW)

    agent.handle_user_message("Selam")

    system = llm.requests[0]["system"]
    assert "Friday, 2026-11-20 10:00" in system
    assert "`appointment_datetime`" in system
    assert llm.requests[0]["tools"][0].name == "update_fields"


def test_update_fields_schema_lists_form_fields() -> None:
    agent = make_agent([])
    schema = agent.tools.specs[0].parameters["properties"]["fields"]

    assert set(schema["properties"]) == {
        "full_name",
        "phone",
        "appointment_datetime",
        "complaint",
    }
    assert schema["additionalProperties"] is False
