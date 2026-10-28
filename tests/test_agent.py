"""Agent verification logic and the tool-calling loop.

The verification tests run offline. The loop test uses a scripted chat model, so it exercises
LangGraph, the tools and the recorder without calling an LLM.
"""

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from expert_finder.agent import AgentResult, _refs_in, run_agent
from expert_finder.app import ExpertFinder
from expert_finder.config import get_settings
from expert_finder.tools import ToolRecorder, build_tools

# --- offline: reference parsing and grounding rules ---------------------------


@pytest.mark.parametrize("text,expected", [
    ("Call them [biz_0042].", ["biz_0042"]),
    ("Two sources [R1][P2] agree.", ["R1", "P2"]),
    ("Both [R1, R2] say so.", ["R1", "R2"]),
    ("Repeated [R1] and again [R1].", ["R1"]),
    ("A bracketed aside [see the note above] is not a citation.", []),
    ("No citations here.", []),
])
def test_ref_parsing(text, expected):
    assert _refs_in(text) == expected


def test_grounded_requires_real_refs_real_phones_and_at_least_one_citation():
    assert AgentResult(answer="a [R1]", mode="graph", cited_refs=["R1"]).grounded

    invented_ref = AgentResult(answer="a [R9]", mode="graph", cited_refs=["R9"], unsupported_refs=["R9"])
    assert not invented_ref.grounded and "R9" in invented_ref.problems[0]

    invented_phone = AgentResult(answer="call (512) 555-1234", mode="graph", cited_refs=["R1"],
                                 fabricated_phones=["(512) 555-1234"])
    assert not invented_phone.grounded and "phone" in invented_phone.problems[0]

    # An answer that cites nothing must not count as grounded just because nothing was wrong.
    says_nothing = AgentResult(answer="Call Hernandez.", mode="graph", uncited=True)
    assert not says_nothing.grounded and "no evidence" in says_nothing.problems[0]


def tools_for(app, mode) -> set[str]:
    return {t.name for t in build_tools(app, mode, ToolRecorder())}


# --- the tool-calling loop, with a scripted model -----------------------------


class ScriptedModel(BaseChatModel):
    """Calls one tool, then answers with the text in `reply`."""

    tool_name: str = "get_business_report"
    tool_args: dict = {"business_name": "Bee Cave Roofing"}
    reply: str = "Be careful with them [biz_0080]."
    bound_tools: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        ScriptedModel.bound_tools = [t.name for t in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if any(m.type == "tool" for m in messages):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.reply))])
        call = {"name": self.tool_name, "id": "1", "args": self.tool_args}
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="", tool_calls=[call]))])


@pytest.fixture(scope="module")
def app():
    s = get_settings()
    db = s.storage_dir / "home_services.db"
    if not db.exists():
        pytest.skip("SQLite store not built; run `expert-finder build-sql`")
    a = ExpertFinder(s)
    yield a
    a.close()


@pytest.fixture
def scripted(app, monkeypatch):
    model = ScriptedModel()
    monkeypatch.setattr(ExpertFinder, "chat_model", lambda self, temperature=0.0: model)
    return model


def test_loop_records_tools_and_accepts_a_supported_citation(app, scripted):
    result = run_agent(app, "Is Bee Cave Roofing legit?", mode="vector", thread_id="t1")
    assert [c["name"] for c in result.tool_calls] == ["get_business_report"]
    assert result.cited_refs == ["biz_0080"] and result.unsupported_refs == []
    assert result.grounded and result.stats["tool_calls"] == 1


def test_loop_flags_an_invented_citation(app, scripted):
    scripted.reply = "Trust them completely [biz_9999]."
    result = run_agent(app, "Is Bee Cave Roofing legit?", mode="vector", thread_id="t2")
    assert result.unsupported_refs == ["biz_9999"] and not result.grounded


def test_loop_flags_an_invented_phone_number(app, scripted):
    scripted.reply = "Call them on (512) 555-9999 [biz_0080]."
    result = run_agent(app, "Is Bee Cave Roofing legit?", mode="vector", thread_id="t3")
    assert result.fabricated_phones == ["(512) 555-9999"] and not result.grounded


def test_loop_accepts_a_phone_number_a_tool_returned(app, scripted):
    scripted.reply = "Their number is (737) 555-0179 [biz_0080]."
    result = run_agent(app, "Is Bee Cave Roofing legit?", mode="vector", thread_id="t4")
    assert result.fabricated_phones == [] and result.grounded


def test_tool_loadout_binds_only_the_selected_tools(app, scripted):
    """The loadout is opt-in, so it needs a test or it quietly rots."""
    from expert_finder.config import get_settings

    if not get_settings().openai_api_key:
        pytest.skip("tool selection embeds tool descriptions, which needs OpenAI")
    run_agent(app, "Who do people in Mueller use for landscaping?", mode="graph",
              thread_id="loadout", use_tool_loadout=True)
    bound = ScriptedModel.bound_tools
    assert len(bound) <= get_settings().tool_loadout_k + 1   # top-k plus the always-on search tool
    assert "search_reviews_and_posts" in bound               # the always-on one
    assert set(bound) < tools_for(app, "graph")


def test_graph_mode_binds_more_tools_than_vector_mode(app, scripted):
    run_agent(app, "x", mode="vector", thread_id="t5")
    vector_bound = set(ScriptedModel.bound_tools)
    run_agent(app, "x", mode="graph", thread_id="t6")
    graph_bound = set(ScriptedModel.bound_tools)
    assert vector_bound == tools_for(app, "vector")
    assert graph_bound == tools_for(app, "graph")
    assert vector_bound < graph_bound
