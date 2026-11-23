"""Simulated-user runner tests: both sides are scripted, no network."""

from __future__ import annotations

import json

from app.tracing import InMemorySink, TracedLLMClient, Tracer
from evals.simulate import DEFAULT_PERSONAS_PATH, END_MARKER, run_persona, simulator_prompt
from tests.test_agent import ScriptedLLM, text_reply, tool_reply

PERSONA = {
    "id": "sim_test",
    "language": "Turkish",
    "persona": "Polite.",
    "goal": {"full_name": "Deniz Yılmaz", "phone": "0532 111 22 33"},
    "expected": {
        "full_name": "Deniz Yılmaz",
        "phone": "0532 111 22 33",
        "appointment_datetime": "2026-11-23 14:30",
        "complaint": {"contains_any": ["hassas"]},
    },
    "expect_submit": True,
}
VALUES = {
    "full_name": "Deniz Yılmaz",
    "phone": "0532 111 22 33",
    "appointment_datetime": "2026-11-23 14:30",
    "complaint": "Diş hassasiyeti",
}


def agent_llm() -> ScriptedLLM:
    return ScriptedLLM(
        [
            tool_reply("update_fields", {"fields": VALUES}),
            text_reply("Özet: Deniz Yılmaz, 23 Kasım 14:30. Onaylıyor musunuz?"),
            tool_reply("submit_form", {}),
            text_reply("Randevunuz alındı."),
            text_reply("Rica ederim, iyi günler!"),
        ]
    )


def test_a_confirmed_booking_passes_and_the_patient_is_not_traced() -> None:
    patient = ScriptedLLM(
        [
            text_reply("Deniz Yılmaz, 0532 111 22 33, pazartesi 14:30, diş hassasiyeti."),
            text_reply("Evet, onaylıyorum."),
            text_reply(f"Teşekkürler! {END_MARKER}"),
        ]
    )
    sink = InMemorySink()
    tracer = Tracer([sink])
    agent = TracedLLMClient(agent_llm(), tracer, provider="fake")  # as in the real run
    result = run_persona(PERSONA, agent, patient, None, tracer)

    assert result["passed"], result["checks"]
    assert result["submitted"] is True
    # The closing message still reaches the agent (it may carry a confirmation).
    assert [t["user"] for t in result["transcript"]] == [
        "Deniz Yılmaz, 0532 111 22 33, pazartesi 14:30, diş hassasiyeti.",
        "Evet, onaylıyorum.",
        "Teşekkürler!",
    ]
    assert result["metrics"]["simulator_llm_calls"] == 3
    # The patient model sees the agent's replies as its "user" turns.
    assert patient.requests[1]["history"][-1]["content"].startswith("Özet:")
    # Only the agent's LLM calls are in the trace.
    assert sum(e["type"] == "llm_call" for e in sink.events) == 5


def test_a_conversation_that_never_ends_fails_the_flow_check() -> None:
    patient = ScriptedLLM([text_reply("Merhaba")] * 2)
    agent = ScriptedLLM([text_reply("Adınız nedir?")] * 2)
    result = run_persona(PERSONA, agent, patient, None, Tracer(), max_user_turns=2)

    flow = next(c for c in result["checks"] if c["kind"] == "flow")
    assert not flow["passed"] and not result["passed"]
    assert len(result["transcript"]) == 2


def test_wrong_values_or_an_unexpected_submit_fail() -> None:
    persona = {**PERSONA, "expect_submit": False}
    patient = ScriptedLLM([text_reply("bilgiler..."), text_reply("Evet."), text_reply(END_MARKER)])
    result = run_persona(persona, agent_llm(), patient, None, Tracer())
    failed = {c["kind"] for c in result["checks"] if not c["passed"]}
    assert failed == {"submit"}


def test_every_persona_is_complete_and_the_prompt_hides_the_expected_values() -> None:
    personas = json.loads(DEFAULT_PERSONAS_PATH.read_text(encoding="utf-8"))["personas"]
    assert len({p["id"] for p in personas}) == len(personas) >= 10
    for persona in personas:
        assert {"full_name", "phone", "appointment_datetime", "complaint"} <= set(
            persona["expected"]
        )
        prompt = simulator_prompt(persona)
        assert persona["persona"] in prompt and END_MARKER in prompt
        # The patient gets its goal in words, never the normalized answer.
        assert persona["expected"]["appointment_datetime"] not in prompt
