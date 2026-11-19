"""Minimal terminal chat loop.

Run with:  python -m app.cli
"""

from __future__ import annotations

import sys

from openai import OpenAIError

from app.llm_client import HistoryItem, create_llm_client
from app.prompts import SYSTEM_PROMPT

EXIT_COMMANDS = {"exit", "quit", "q", "çıkış"}


def main() -> int:
    try:
        client = create_llm_client()
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    history: list[HistoryItem] = []
    print(f"Form agent ({client.model}). Type 'exit' to quit.\n")

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

        history.append(client.user_message(user_input))
        try:
            response = client.chat(history, system=SYSTEM_PROMPT)
        except OpenAIError as error:
            # Drop the unanswered message so history stays consistent, then let the user retry.
            history.pop()
            print(f"[LLM error] {error}\n", file=sys.stderr)
            continue

        history.extend(response.history_items)
        print(f"Agent: {response.text}")
        print(
            f"  [{response.model} | in={response.input_tokens} out={response.output_tokens} "
            f"tokens | {response.latency_ms:.0f} ms]\n"
        )


if __name__ == "__main__":
    raise SystemExit(main())
