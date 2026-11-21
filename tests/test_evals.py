"""Tests for the eval scoring logic, so a wrong number in the README can't come from a bug here."""

from __future__ import annotations

import json

from app.pricing import estimate_cost_usd
from evals.run_evals import (
    DEFAULT_CASES_PATH,
    check_fields,
    check_reply,
    evaluate_expectations,
    fold_text,
    summarize,
)


def test_field_checks_exact_contains_and_absent() -> None:
    values = {"full_name": "Ali Veli", "complaint": "Sol alt diş ağrısı"}

    results = check_fields(
        {
            "full_name": "Ali Veli",
            "complaint": {"contains_any": ["AĞRI", "sızı"]},
            "phone": None,
            "appointment_datetime": "2026-11-23 11:00",
        },
        values,
        turn=None,
    )

    assert [r.passed for r in results] == [True, True, True, False]
    assert all(r.kind == "field" for r in results)


def test_reply_groups_need_one_match_each() -> None:
    expect = {"reply_contains": [["24 saat", "24 hours"], ["200"]], "reply_not_contains": ["TL"]}

    passing = check_reply(expect, "24 saat öncesine kadar ücretsiz, sonra 200 lira.", "rag", 0)
    failing = check_reply(expect, "24 saat öncesine kadar ücretsiz, sonra 200 TL.", "rag", 0)

    assert all(r.passed for r in passing)
    assert [r.passed for r in failing] == [True, True, False]


def test_reply_kind_depends_on_category() -> None:
    expect = {"reply_contains": [["ok"]]}
    common = {"values": {}, "reply": "ok", "tools_called": [], "submitted": False, "turn": 0}

    assert evaluate_expectations(expect, category="rag_in_doc", **common)[0].kind == "rag"
    assert evaluate_expectations(expect, category="normal", **common)[0].kind == "reply"


def test_submit_and_tool_checks() -> None:
    results = evaluate_expectations(
        {"submitted": False, "tools_called": ["search_knowledge_base"]},
        category="no_confirmation",
        values={},
        reply="",
        tools_called=["update_fields"],
        submitted=True,
        turn=None,
    )

    assert {(r.kind, r.passed) for r in results} == {("tool", False), ("submit", False)}


def _result(category, passed, expected_submit, submitted, checks):
    return {
        "id": category,
        "category": category,
        "passed": passed,
        "error": None,
        "expected_submit": expected_submit,
        "submitted": submitted,
        "checks": checks,
        "metrics": {"user_turns": 2, "llm_calls": 4, "latency_ms": 3000, "cost_usd": 0.001},
    }


def test_summary_rates() -> None:
    field_ok = {"kind": "field", "passed": True}
    field_bad = {"kind": "field", "passed": False}
    results = [
        _result("normal", True, True, True, [field_ok, field_ok]),
        _result("normal", False, True, False, [field_bad]),
        _result("prompt_injection", False, False, True, [field_ok]),
        _result("no_confirmation", True, False, False, []),
    ]

    summary = summarize(results)

    assert summary["cases"] == {"passed": 2, "total": 4, "rate": 0.5}
    assert summary["field_accuracy"]["rate"] == 0.75
    assert summary["wrong_submit_rate"] == {"passed": 1, "total": 2, "rate": 0.5}
    assert summary["missed_submit_rate"] == {"passed": 1, "total": 2, "rate": 0.5}
    assert summary["rag_accuracy"]["total"] == 0
    assert summary["by_category"]["normal"]["total"] == 2
    assert summary["avg_turn_latency_s"] == 1.5


def test_case_clock_defaults_to_friday_and_can_be_overridden() -> None:
    from evals.run_evals import EVAL_NOW, case_clock

    assert case_clock({}) == EVAL_NOW
    saturday = case_clock({"now": "2026-11-21T10:00"})
    assert (saturday.weekday(), saturday.hour, saturday.tzinfo) == (5, 10, EVAL_NOW.tzinfo)


def test_cases_file_is_well_formed() -> None:
    cases = json.loads(DEFAULT_CASES_PATH.read_text(encoding="utf-8"))["cases"]

    assert 20 <= len(cases) <= 30
    assert len({c["id"] for c in cases}) == len(cases)
    for case in cases:
        assert case["turns"], case["id"]
        assert "submitted" in case.get("final", {}), case["id"]


def test_cost_estimate() -> None:
    # 1M uncached input + 1M cached + 1M output on deepseek-flash = 0.30 + 0.006 + 1.20
    assert estimate_cost_usd("deepseek-flash", 2_000_000, 1_000_000, 1_000_000) == 1.506
    assert estimate_cost_usd("unknown-model", 10, 0, 10) is None


def test_price_pattern_ignores_tl_inside_words() -> None:
    expect = {"reply_not_matches": [r"\d[\d.,]*\s*(TL|₺|lira)"]}

    no_price = check_reply(expect, "Ortodonti fiyatları hakkında bilgim yok.", "rag", 0)
    price = check_reply(expect, "Tel tedavisi 45.000 TL'dir.", "rag", 0)

    assert no_price[0].passed is True
    assert price[0].passed is False


def test_fold_text_handles_turkish_i() -> None:
    assert fold_text("İngilizce") == "ingilizce"
    assert fold_text("AĞRI") == "ağrı"
    assert "İngilizce".casefold() != "ingilizce"  # why plain casefold is not enough
