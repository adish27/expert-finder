"""The conversational agent, in vector and graph modes.

Built with LangGraph's create_agent. Conversation memory comes from a checkpointer keyed by
thread_id, so follow-up turns ("what about the second one?") work without managing history here.
After each run, every ref cited in the answer is checked against the refs the tools actually
returned; anything else is reported as an unsupported citation.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from expert_finder.app import ExpertFinder, Mode
from expert_finder.tools import GRAPH_TOOLS, SHARED_TOOLS, ToolRecorder, build_tools

SYSTEM_PROMPT = """You are Expert Finder, an assistant that helps homeowners in the Austin, Texas area \
find the right professional for plumbing, electrical, HVAC, roofing and landscaping work.

How to behave:
- Answer like a well-informed neighbour: direct, concrete, and honest about what you do not know.
- Always use tools to get facts. Never invent a business, person, price, phone number or review.
- Recommend specific businesses by name, say why, and give the phone number when you have it.
- Quote real numbers for prices, and say what the comparison is based on (for example "median of 12 \
real quotes"), not a vague range.
- Cite the evidence for every factual claim, using the bracketed identifiers exactly as the tools \
print them: [R1], [P2], [G3], [biz_0042], [rev_000123], [exp_0051], [post_00194]. Put each one right \
after the claim it supports. Every business you recommend must carry its [biz_...] identifier. An \
answer with no citations is not acceptable when the tools returned any.
- Say plainly when the evidence is thin, based on very few jobs, or contradictory. A conclusion \
drawn from two jobs is not the same as one drawn from twenty, so say which it is.
- Never state a phone number, price, rating or name that a tool did not return. If you do not have \
a phone number, say to look it up rather than producing one.
- When the job is urgent, lead with businesses that answer and arrive quickly, not just the \
best-rated one. Note that 24/7 emergency service and routine scheduling are different things: a \
company can take emergency calls tonight and still be weeks out for booked work. Mention the \
after-hours surcharge when the user is calling outside normal hours.

Red flags you must surface when you see them, even if the user did not ask:
- final bills coming in well above the written quote
- unresolved complaints, or an owner who argues with reviewers instead of fixing problems
- an expired license in a trade that requires one (plumbing, electrical and HVAC require a Texas \
license; roofing and landscaping do not, so no license on file is normal for those)
- a rating propped up by a burst of reviews from brand-new or untraceable accounts
- no general liability insurance

When the request is missing something you need, ask one short follow-up question rather than \
guessing: which city or neighbourhood, repair or replacement, and the budget if it matters.
Keep answers short: the recommendation, the reason with citations, and how to reach them."""

REF_PATTERN = re.compile(r"\[([^\[\]]{1,80})\]")
PHONE_PATTERN = re.compile(r"\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}")


@dataclass
class AgentResult:
    answer: str
    mode: Mode
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    cited_refs: list[str] = field(default_factory=list)
    unsupported_refs: list[str] = field(default_factory=list)
    fabricated_phones: list[str] = field(default_factory=list)
    uncited: bool = False
    evidence: list = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def grounded(self) -> bool:
        """No invented citations, no invented contact details, and at least one citation.

        Checking only unsupported refs would pass an answer that cites nothing at all, so an
        answer with evidence available but no citations counts as ungrounded too.
        """
        return not self.unsupported_refs and not self.fabricated_phones and not self.uncited

    @property
    def problems(self) -> list[str]:
        out = []
        if self.unsupported_refs:
            out.append(f"cites references no tool returned: {', '.join(self.unsupported_refs)}")
        if self.fabricated_phones:
            out.append(f"states phone numbers no tool returned: {', '.join(self.fabricated_phones)}")
        if self.uncited:
            out.append("cites no evidence although the tools returned some")
        return out


def _refs_in(text: str) -> list[str]:
    """Bracketed references, allowing [R1], [R1, R2] and [R1][R2]."""
    out = []
    for group in REF_PATTERN.findall(text):
        for part in group.split(","):
            ref = part.strip()
            # A citation is a short token, not a sentence: this skips ordinary bracketed prose.
            if ref and " " not in ref and len(ref) <= 40:
                out.append(ref)
    return list(dict.fromkeys(out))


def _digits(phone: str) -> str:
    return re.sub(r"\D", "", phone)


def tool_names(mode: Mode) -> list[str]:
    return SHARED_TOOLS + (GRAPH_TOOLS if mode == "graph" else [])


def build_agent(app: ExpertFinder, mode: Mode, use_tool_loadout: bool = False,
                checkpointer: Any | None = None):
    """Create the agent for one mode.

    `use_tool_loadout` binds only the top-k semantically relevant tools per turn. With nine tools
    it is not needed, and it adds variance, so it stays off for evaluation runs; it is exposed for
    demonstrating the pattern and for when the tool catalogue grows.
    """
    from langchain.agents import create_agent

    recorder = ToolRecorder()
    tools = build_tools(app, mode, recorder)
    middleware = []
    if use_tool_loadout:
        from expert_finder.tool_loadout import ToolLoadout

        loadout = ToolLoadout(tools, app.index.embeddings, k=app.settings.tool_loadout_k,
                              always=["search_reviews_and_posts"])
        middleware.append(loadout.middleware())
    agent = create_agent(app.chat_model(), tools, system_prompt=SYSTEM_PROMPT, middleware=middleware,
                         checkpointer=checkpointer or InMemorySaver())
    return agent, recorder


def run_agent(app: ExpertFinder, question: str, mode: Mode = "graph", thread_id: str = "cli",
              agent=None, recorder: ToolRecorder | None = None, **kwargs) -> AgentResult:
    """Ask one question. Pass an existing agent and recorder to continue a conversation."""
    if agent is None:
        agent, recorder = build_agent(app, mode, **kwargs)
    assert recorder is not None
    recorder.reset()

    started = time.perf_counter()
    state = agent.invoke({"messages": [{"role": "user", "content": question}]},
                         config={"configurable": {"thread_id": thread_id}})
    elapsed = round((time.perf_counter() - started) * 1000)

    messages = state["messages"]
    final = next(m for m in reversed(messages) if isinstance(m, AIMessage) and m.text.strip())
    # Everything the tools actually produced this turn, for verifying the answer against it.
    available: set[str] = set()
    available_phones: set[str] = set()
    for m in messages:
        if isinstance(m, ToolMessage):
            available |= set(_refs_in(m.text))
            available_phones |= {_digits(p) for p in PHONE_PATTERN.findall(m.text)}
    cited = _refs_in(final.text)
    # A wrong phone number is the most damaging thing this app could invent, so check it directly
    # rather than trusting the prompt rule.
    fabricated_phones = [p for p in PHONE_PATTERN.findall(final.text) if _digits(p) not in available_phones]

    usage = {}
    for m in messages:
        if isinstance(m, AIMessage) and m.usage_metadata:
            for key in ("input_tokens", "output_tokens"):
                usage[key] = usage.get(key, 0) + m.usage_metadata.get(key, 0)

    return AgentResult(
        answer=final.text, mode=mode,
        tool_calls=[{"name": c.name, "args": c.args, "output": c.output} for c in recorder.calls],
        cited_refs=cited,
        unsupported_refs=[r for r in cited if r not in available],
        fabricated_phones=list(dict.fromkeys(fabricated_phones)),
        uncited=bool(available and not cited),
        evidence=recorder.evidence_items(),
        stats={"latency_ms": elapsed, "tool_calls": len(recorder.calls),
               "evidence_items": len(recorder.evidence_items()), **usage,
               "graph_items": sum(1 for i in recorder.evidence_items() if i.via == "graph")})
