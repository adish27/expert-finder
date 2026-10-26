"""Chat page: ask about a home project and get a grounded recommendation."""

from __future__ import annotations

import uuid

import streamlit as st
from shared import ask, check_stores, render_evidence, render_grounding, render_tool_trace, sidebar

st.set_page_config(page_title="Home Services Expert Finder", page_icon="🔧", layout="wide")

EXAMPLES = [
    "My water heater is leaking in the garage. Who can come out today?",
    "Is Bee Cave Roofing legit? I'm seeing a lot of 5-star reviews.",
    "I was quoted $21,000 for a new roof, about 28 squares. Is that fair?",
    "Who do my neighbours use for sprinkler repair?",
    "Who's the best electrician for a Tesla charger install?",
]

st.title("🔧 Home Services Expert Finder")
st.caption("Plumbing, electrical, HVAC, roofing and landscaping in the Austin, TX area. "
           "Every claim is cited and checked against the retrieved evidence. All data is synthetic.")

problems = check_stores()
if problems:
    st.error("Some stores are not ready:\n\n"
             + "\n".join(f"- **{k}**: {v}" for k, v in problems.items()))
    st.stop()

ctx = sidebar()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.divider()
    if st.button("New conversation", use_container_width=True):
        # A new thread id starts fresh memory; the agents themselves can be reused.
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.history = []
        st.rerun()

# Example prompts, shown only before the first question.
pending = None
if not st.session_state.history:
    st.markdown("**Try one of these:**")
    columns = st.columns(len(EXAMPLES))
    for column, example in zip(columns, EXAMPLES, strict=True):
        if column.button(example, use_container_width=True):
            pending = example

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("result"):
            result = turn["result"]
            render_grounding(result)
            with st.expander(f"Evidence ({len(result.evidence)} items)"):
                render_evidence(result)
            with st.expander("What the agent did"):
                render_tool_trace(result)

question = st.chat_input("Describe the problem, or ask about a company or a price...") or pending

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner(f"Searching in {ctx['mode']} mode..."):
            try:
                result = ask(question, ctx, st.session_state.thread_id)
            except Exception as e:  # keep the chat usable instead of showing a stack trace
                st.error(f"Something went wrong: {type(e).__name__}: {e}")
                st.stop()
        st.markdown(result.answer)
        render_grounding(result)
        with st.expander(f"Evidence ({len(result.evidence)} items)"):
            render_evidence(result)
        with st.expander("What the agent did"):
            render_tool_trace(result)
    st.session_state.history.append({"role": "assistant", "content": result.answer, "result": result})
