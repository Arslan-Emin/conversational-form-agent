"""Terminal chat with the form agent.

Run with:  python -m app.cli
Add --debug to see every tool call and its result.
"""

from __future__ import annotations

import argparse
import json
import sys

from openai import OpenAIError

from app.agent import AgentTurn, FormAgent
from app.llm_client import create_llm_client
from app.state import FormState

EXIT_COMMANDS = {"exit", "quit", "q", "çıkış"}


def format_state(state: FormState) -> str:
    lines = []
    for field in state.fields:
        value = state.values.get(field.name)
        mark = "x" if value else " "
        lines.append(f"    [{mark}] {field.name}: {value or '-'}")
    return "\n".join(lines)


def print_turn(turn: AgentTurn, state: FormState, debug: bool) -> None:
    if debug:
        for call in turn.tool_calls:
            args = json.dumps(call.arguments, ensure_ascii=False)
            result = json.dumps(call.result, ensure_ascii=False)
            print(f"  [tool] {call.name}({args})\n         -> {result}")
    print(f"Agent: {turn.reply}")
    print(
        f"  [{turn.llm_calls} LLM call(s) | in={turn.input_tokens} out={turn.output_tokens} "
        f"tokens | {turn.latency_ms:.0f} ms]"
    )
    print("  Form state:")
    print(format_state(state) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chat with the form agent.")
    parser.add_argument("--debug", action="store_true", help="show tool calls and results")
    args = parser.parse_args(argv)

    try:
        llm = create_llm_client()
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    agent = FormAgent(llm)
    print(f"Form agent ({llm.model}). Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not user_input:
            continue
        if user_input.lower() in EXIT_COMMANDS:
            return 0

        try:
            turn = agent.handle_user_message(user_input)
        except OpenAIError as error:
            print(f"[LLM error] {error}\n", file=sys.stderr)
            continue
        print_turn(turn, agent.state, args.debug)


if __name__ == "__main__":
    raise SystemExit(main())
