"""Chat page: ask about a home project and get a grounded recommendation."""

from __future__ import annotations

import uuid

import streamlit as st
from shared import ask, check_stores, render_evidence, render_grounding, render_tool_trace, sidebar

st.set_page_config(page_title="Home Services Expert Finder", page_icon="🔧", layout="wide")

# One question per tool, grouped by how the answer is found. Each was checked to route to the
# tool it is meant to demonstrate.
EXAMPLE_GROUPS = {
    "Semantic search over text": [
        ("What do people say about plumbers pushing work you don't need?", "search_reviews_and_posts"),
        ("Who installs tankless water heaters and speaks Spanish?", "search_business_profiles"),
    ],
    "Structured lookups (SQL)": [
        ("My AC died and it's 104 out in Round Rock. Who can come today?", "find_businesses"),
        ("What should a new AC system cost in Round Rock?", "get_price_range"),
        ("A roofer quoted me $21,000 for 28 squares. Am I being overcharged?", "check_quote"),
        ("Has Hill Country Air got worse recently?", "get_business_report"),
    ],
    "Graph traversal (Neo4j)": [
        ("Who do people in Mueller use for landscaping?", "neighbor_recommendations"),
        ("Who should I ask for at Cool Capitol City HVAC?", "who_to_ask_for"),
        ("Are Bee Cave Roofing's 5-star reviews real? Something feels off.", "check_review_authenticity"),
    ],
}

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

# Example prompts, shown only before the first question. Grouped so the pills also show which
# part of the system answers each kind of question.
pending = None
if not st.session_state.history:
    st.markdown("**Try one of these** — each one exercises a different part of the system:")
    for column, (group, examples) in zip(st.columns(len(EXAMPLE_GROUPS)), EXAMPLE_GROUPS.items(), strict=True):
        with column:
            st.caption(f"**{group}**")
            for example, tool in examples:
                if st.button(example, use_container_width=True, key=f"ex_{tool}",
                             help=f"Expected to use the `{tool}` tool"):
                    pending = example
    if ctx["mode"] == "vector":
        st.caption("In vector mode the three graph questions still get answered, but without the "
                   "graph tools — a good way to see what the graph adds.")

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
