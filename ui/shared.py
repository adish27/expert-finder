"""Shared Streamlit helpers: cached resources and rendering.

Streamlit reruns the whole script on every interaction, so anything expensive must be cached.
`@st.cache_resource` keeps one ExpertFinder (and therefore one Neo4j driver and Pinecone client)
per server process; SQLite connections are thread-local, which matches Streamlit running each
session in its own thread.
"""

from __future__ import annotations

import streamlit as st

from expert_finder.agent import build_agent, run_agent
from expert_finder.app import ExpertFinder
from expert_finder.config import get_settings

CITIES = ["Austin", "Round Rock", "Cedar Park", "Pflugerville", "Georgetown", "Leander", "Kyle",
          "Buda", "Lakeway", "Dripping Springs"]

MODE_HELP = {
    "graph": "Vector search, then expansion through the Neo4j graph: resolves who posts are talking "
             "about, pulls in whole discussion threads, and adds three graph-only tools.",
    "vector": "Semantic search over Pinecone only, with the shared structured tools.",
}

KIND_LABEL = {"business": "Businesses", "expert": "People", "review": "Reviews",
              "post": "Social posts", "thread": "Rest of the thread", "graph": "Graph facts"}
KIND_ORDER = ["business", "expert", "review", "post", "thread", "graph"]
VIA_LABEL = {"vector": "semantic search", "graph": "graph traversal", "sql": "structured query"}


@st.cache_resource(show_spinner="Connecting to Pinecone, Neo4j and SQLite...")
def get_app() -> ExpertFinder:
    """One container per server process, reused across reruns and sessions."""
    return ExpertFinder()


@st.cache_data(ttl=60, show_spinner=False)
def check_stores() -> dict[str, str]:
    """Report anything missing, so the UI can explain rather than throw.

    Cached briefly rather than for the process lifetime: building a missing store should clear
    the warning on the next interaction, not require restarting the server.
    """
    app = get_app()
    problems = {}
    s = get_settings()
    if not s.openai_api_key:
        problems["OpenAI"] = "OPENAI_API_KEY is not set in .env"
    if not (s.storage_dir / "home_services.db").exists():
        problems["SQLite"] = "not built - run `uv run expert-finder build-sql`"
    try:
        if not app.index.exists():
            problems["Pinecone"] = "index not built - run `uv run expert-finder index`"
    except Exception as e:
        problems["Pinecone"] = f"{type(e).__name__}: {e}"
    try:
        app.driver.verify_connectivity()
    except Exception as e:
        problems["Neo4j"] = f"{type(e).__name__}: {e}"
    return problems


def sidebar() -> dict:
    """Mode and location controls. Location is added to the question as context."""
    with st.sidebar:
        st.subheader("Retrieval mode")
        mode = st.radio("mode", ["graph", "vector"], format_func=str.title, label_visibility="collapsed",
                        key="mode")
        st.caption(MODE_HELP[mode])
        st.divider()
        st.subheader("Your location")
        city = st.selectbox("City", CITIES, index=0)
        neighborhood = st.text_input("Neighborhood (optional)", placeholder="Hyde Park")
        st.divider()
        model = get_settings().chat_model
        st.caption(f"Model: `{model}`  \nEvery answer is checked against what the tools returned.")
    return {"mode": mode, "city": city, "neighborhood": neighborhood}


def with_context(question: str, ctx: dict) -> str:
    """Give the agent the user's location so it does not have to ask."""
    where = f"{ctx['neighborhood']}, {ctx['city']}" if ctx["neighborhood"] else ctx["city"]
    return f"{question}\n\n(I live in {where}, TX.)"


def session_agent(mode: str):
    """Build the agent once per session and mode, so conversation memory survives reruns."""
    key = f"agent_{mode}"
    if key not in st.session_state:
        st.session_state[key] = build_agent(get_app(), mode)
    return st.session_state[key]


def ask(question: str, ctx: dict, thread_id: str):
    agent, recorder = session_agent(ctx["mode"])
    return run_agent(get_app(), with_context(question, ctx), mode=ctx["mode"], thread_id=thread_id,
                     agent=agent, recorder=recorder)


# --- rendering ---------------------------------------------------------------


def render_grounding(result) -> None:
    if result.grounded:
        st.success(f"Grounded: {len(result.cited_refs)} citations, all traced to retrieved evidence.", icon="✅")
    else:
        st.warning("Grounding check failed — " + "; ".join(result.problems), icon="⚠️")


def render_evidence(result) -> None:
    """Evidence cards, grouped by type, labelled with the refs the answer can cite."""
    if not result.evidence:
        st.caption("This answer came from structured lookups rather than retrieved text; "
                   "see 'What the agent did' for the raw tool output.")
        return
    cited = set(result.cited_refs)
    for kind in KIND_ORDER:
        group = [e for e in result.evidence if e.kind == kind]
        if not group:
            continue
        st.markdown(f"**{KIND_LABEL[kind]}** ({len(group)})")
        for e in group:
            used = "✓ cited" if e.ref in cited else ""
            source = e.source
            label = (source.get("business_name") or source.get("community")
                     or source.get("platform") or e.kind)
            meta = " · ".join(str(x) for x in [source.get("platform"), label, source.get("created_date"),
                                               f"score {e.score}" if e.score else None,
                                               VIA_LABEL.get(e.via, e.via)] if x)
            with st.expander(f"`[{e.ref}]` {meta} {used}"):
                st.write(e.text)
                caption = f"id: `{source.get('id')}`"
                if source.get("relation"):
                    caption += f" · {source['relation'].replace('_', ' ')}"
                st.caption(caption)


def render_tool_trace(result) -> None:
    """Each tool call with its arguments and full output.

    Price and quote tools return no record ids to cite, so their numbers only appear here.
    """
    for call in result.tool_calls:
        args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items() if v is not None)
        st.markdown(f"**`{call['name']}({args})`**")
        st.code(call.get("output", "") or "(no output)", language=None, wrap_lines=True)
    st.caption(" · ".join(f"{k}: {v}" for k, v in result.stats.items()))
