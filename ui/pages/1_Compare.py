"""Compare page: the same question answered by vector RAG and graph RAG, side by side."""

from __future__ import annotations

import streamlit as st
from shared import (
    CITIES,
    check_stores,
    get_app,
    render_evidence,
    render_grounding,
    render_tool_trace,
    with_context,
)

from expert_finder.agent import run_agent

st.set_page_config(page_title="Vector vs Graph RAG", page_icon="⚖️", layout="wide")

QUESTIONS = [
    "Who do my neighbours use for sprinkler repair?",
    "Is Bee Cave Roofing legit? I'm seeing a lot of 5-star reviews.",
    "Has Hill Country Air got worse recently?",
    "Who's the best person to ask for at Cool Capitol City HVAC?",
    "My water heater is leaking in the garage. Who can come out today?",
]

st.title("⚖️ Vector RAG vs Graph RAG")
st.markdown(
    "Both sides use the same model, the same prompt, the same Pinecone index and the same "
    "structured SQL tools. **Graph mode adds** expansion of each vector hit through Neo4j "
    "(resolving who posts refer to, pulling in whole threads, surfacing businesses the vector "
    "search missed) and three graph-only tools. The difference in the answers is what the graph buys.")

if problems := check_stores():
    st.error("Some stores are not ready:\n\n" + "\n".join(f"- **{k}**: {v}" for k, v in problems.items()))
    st.stop()

with st.form("compare"):
    question = st.selectbox("Question", QUESTIONS, index=0,
                            help="Pick one, or choose 'Custom' below to type your own")
    custom = st.text_input("Or ask your own")
    city = st.selectbox("City", CITIES, index=0)
    neighborhood = st.text_input("Neighborhood (optional)", placeholder="Tarrytown")
    submitted = st.form_submit_button("Run both", type="primary")

if submitted:
    asked = custom.strip() or question
    ctx = {"city": city, "neighborhood": neighborhood}
    app = get_app()
    results = {}
    for mode in ("vector", "graph"):
        with st.spinner(f"Running {mode} mode..."):
            # A fresh thread per run, so neither mode sees the other's conversation.
            results[mode] = run_agent(app, with_context(asked, ctx), mode=mode,
                                      thread_id=f"compare-{mode}-{abs(hash(asked))}")

    st.divider()
    st.subheader(f"“{asked}”")
    summary = {
        "Evidence items": (results["vector"].stats["evidence_items"], results["graph"].stats["evidence_items"]),
        "Of which from the graph": (0, results["graph"].stats["graph_items"]),
        "Tool calls": (results["vector"].stats["tool_calls"], results["graph"].stats["tool_calls"]),
        "Citations": (len(results["vector"].cited_refs), len(results["graph"].cited_refs)),
        "Latency (ms)": (results["vector"].stats["latency_ms"], results["graph"].stats["latency_ms"]),
        "Input tokens": (results["vector"].stats.get("input_tokens", 0), results["graph"].stats.get("input_tokens", 0)),
        "Grounded": (results["vector"].grounded, results["graph"].grounded),
    }
    st.dataframe({"Metric": list(summary), "Vector": [v[0] for v in summary.values()],
                  "Graph": [v[1] for v in summary.values()]}, hide_index=True, use_container_width=True)

    for column, mode in zip(st.columns(2), ("vector", "graph"), strict=True):
        result = results[mode]
        with column:
            st.markdown(f"### {mode.title()} RAG")
            st.markdown(result.answer)
            render_grounding(result)
            with st.expander(f"Evidence ({len(result.evidence)} items)"):
                render_evidence(result)
            with st.expander("What the agent did"):
                render_tool_trace(result)
