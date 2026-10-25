"""Tool loadout: semantically select the top-k tools for a query before each model call.

Tool descriptions are embedded once; for every model call the latest user message is
matched against them and only the best k tools (plus any `always` tools) are bound.
This keeps prompts small and routing focused as the tool catalog grows.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langchain_core.vectorstores import InMemoryVectorStore


class ToolLoadout:
    def __init__(self, tools: list[BaseTool], embeddings: Embeddings, k: int = 3, always: Iterable[str] = ()):
        self.tools = {t.name: t for t in tools}
        self.k = k
        self.always = list(always)
        self._index = InMemoryVectorStore(embeddings)
        self._index.add_documents([Document(page_content=f"{t.name.replace('_', ' ')}: {t.description}",
                                            metadata={"name": t.name}) for t in tools])

    def select(self, query: str) -> list[BaseTool]:
        hits = self._index.similarity_search(query, k=self.k)
        names = list(dict.fromkeys([*self.always, *(h.metadata["name"] for h in hits)]))
        return [self.tools[n] for n in names]

    def middleware(self):
        loadout = self

        @wrap_model_call
        def select_tools(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
            # Select on the latest user turn so the loadout stays stable across tool-calling steps.
            query = next((m.text for m in reversed(request.messages) if isinstance(m, HumanMessage)), "")
            names = {t.name for t in loadout.select(query)}
            return handler(request.override(tools=[t for t in request.tools if getattr(t, "name", None) in names]))

        return select_tools
