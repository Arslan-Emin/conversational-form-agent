"""Streamlit demo: chat with the form agent and watch the form fill itself in.

Run with:  streamlit run streamlit_app.py
"""

from __future__ import annotations

import json

import streamlit as st
from openai import OpenAIError

from app.agent import AgentTurn, FormAgent
from app.factory import FormSetup, create_traced_llm, create_tracer, load_form_setup
from app.jotform import JotformError
from app.llm_client import LLMClient
from app.rag import KnowledgeBase
from app.tracing import Tracer

EXAMPLE_PROMPTS = [
    "Merhaba, yarın öğleden sonra dişim için randevu almak istiyorum",
    "İptal edersem ücret alınır mı?",
    "Hi, I'd like to book a check-up next Monday at 10:00",
]

st.set_page_config(page_title="Form Agent", page_icon="🦷", layout="wide")


# --------------------------------------------------------------------------- #
# Shared resources: created once per server process, reused by every session.
# --------------------------------------------------------------------------- #


@st.cache_resource
def get_tracer() -> Tracer:
    return create_tracer()


@st.cache_resource(show_spinner="Connecting to the LLM...")
def get_llm() -> LLMClient:
    return create_traced_llm(tracer=get_tracer())


@st.cache_resource(show_spinner="Loading the form...", ttl=600)
def get_form_setup() -> FormSetup:
    return load_form_setup()


@st.cache_resource(show_spinner="Loading FAQ knowledge base (first run downloads a model)...")
def get_knowledge_base() -> KnowledgeBase:
    return KnowledgeBase.from_markdown_file()


# --------------------------------------------------------------------------- #
# Per-session state: each browser tab gets its own agent and form state.
# --------------------------------------------------------------------------- #


def start_conversation() -> None:
    setup = get_form_setup()
    st.session_state.agent = FormAgent(
        get_llm(),
        setup.new_state(),
        submitter=setup.create_submitter(live=False),
        knowledge_base=get_knowledge_base(),
        tracer=get_tracer(),
    )
    st.session_state.messages = []  # [{"role", "content", "turn"}]


def render_tool_calls(turn: AgentTurn) -> None:
    with st.expander(
        f"🔧 {len(turn.tool_calls)} tool call(s) · {turn.llm_calls} LLM call(s) · "
        f"{turn.input_tokens + turn.output_tokens} tokens · {turn.latency_ms / 1000:.1f}s"
    ):
        if not turn.tool_calls:
            st.caption("No tools were called.")
        for call in turn.tool_calls:
            status = "✅" if call.result.get("ok") else "❌"
            st.markdown(f"{status} **`{call.name}`**")
            st.code(json.dumps(call.arguments, ensure_ascii=False, indent=2), language="json")
            st.code(json.dumps(call.result, ensure_ascii=False, indent=2), language="json")


def render_sidebar(agent: FormAgent, setup: FormSetup) -> bool:
    """Draw the live form panel. Returns whether tool details should be shown."""
    state = agent.state
    with st.sidebar:
        st.header("📋 Form state")
        st.caption(f"Source: {setup.description}")

        filled = sum(1 for field in state.fields if field.name in state.values)
        st.progress(filled / len(state.fields), text=f"{filled}/{len(state.fields)} fields")

        for field in state.fields:
            value = state.values.get(field.name)
            icon = "✅" if value else ("⬜" if field.required else "▫️")
            st.markdown(f"{icon} **{field.label}**  \n`{value or '—'}`")

        st.caption("Only values that passed validation appear here.")

        if state.submitted:
            if state.submission_id:
                st.success(f"Submitted to Jotform · ID {state.submission_id}")
            else:
                st.info("Submitted (dry-run: nothing was sent)")

        st.divider()
        st.header("⚙️ Settings")
        live = st.toggle(
            "Send to Jotform (live)",
            value=False,
            disabled=not setup.can_submit_live or state.submitted,
            help="Off = dry-run: the submission payload is built but not sent.",
        )
        agent.tools.submitter = setup.create_submitter(live=live)
        show_tools = st.toggle("Show tool calls", value=True)

        turns = [m["turn"] for m in st.session_state.messages if m.get("turn")]
        if turns:
            tokens = sum(t.input_tokens + t.output_tokens for t in turns)
            latency = sum(t.latency_ms for t in turns) / len(turns) / 1000
            col1, col2 = st.columns(2)
            col1.metric("Tokens", f"{tokens:,}")
            col2.metric("Avg. turn", f"{latency:.1f}s")

        if st.button("🔄 New conversation", use_container_width=True):
            start_conversation()
            st.rerun()
    return show_tools


def handle_message(text: str) -> bool:
    """Run one agent turn. Returns False (and shows an error) if the LLM call failed."""
    st.session_state.messages.append({"role": "user", "content": text})
    try:
        with st.spinner("Thinking..."):
            turn = st.session_state.agent.handle_user_message(text)
    except (OpenAIError, JotformError) as error:
        st.session_state.messages.pop()
        st.error(f"Something went wrong, please try again. ({error.__class__.__name__})")
        return False
    st.session_state.messages.append({"role": "assistant", "content": turn.reply, "turn": turn})
    return True


def main() -> None:
    try:
        setup = get_form_setup()
        if "agent" not in st.session_state:
            start_conversation()
    except (RuntimeError, JotformError) as error:
        st.error(f"Setup error: {error}")
        st.stop()

    agent: FormAgent = st.session_state.agent
    st.title("🦷 Dental clinic booking agent")
    st.caption(
        "Book an appointment by chatting instead of filling in a form. "
        "Ask about prices, hours or cancellation at any time. Demo data only."
    )

    clicked = None
    if not st.session_state.messages:
        st.markdown("**Try one of these:**")
        columns = st.columns(len(EXAMPLE_PROMPTS))
        for column, prompt in zip(columns, EXAMPLE_PROMPTS, strict=True):
            if column.button(prompt, use_container_width=True):
                clicked = prompt

    typed = st.chat_input("Type your message...", disabled=agent.state.submitted)
    if (message := (typed or clicked)) and handle_message(message):
        st.rerun()  # redraw from the top so the sidebar shows the updated form state

    show_tools = render_sidebar(agent, setup)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if show_tools and message.get("turn"):
                render_tool_calls(message["turn"])

    if agent.state.submitted:
        st.success("This conversation is complete. Start a new one from the sidebar.")


main()
