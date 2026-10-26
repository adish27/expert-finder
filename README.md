# Home Services Expert Finder

Helps homeowners find the right plumber, electrician, HVAC tech, roofer or landscaper, at a fair
price, backed by evidence from reviews and neighbors. A conversational agent (LangChain + LangGraph,
OpenAI) over Pinecone (semantic search) and Neo4j (relationships), traced in LangSmith. It compares
vector RAG against graph RAG on the same questions. See [docs/application_overview.md](docs/application_overview.md).

**Status:** dataset, Pinecone index, SQLite store, Neo4j graph, retrieval layer, agent and Streamlit UI are built. Evaluation against ground truth is next.

## Layout

```
data/home_services/        synthetic dataset (JSONL + reference JSON); see its README for the data dictionary
docs/                      application overview
scripts/                   dataset catalog + deterministic generator
src/expert_finder/
  config.py          settings from .env (OpenAI, Pinecone, Neo4j, LangSmith)
  dataset.py         loads the raw dataset (never ground_truth/)
  documents.py       records -> documents with metadata (reviews, social posts, business profiles)
  vector_store.py    Pinecone index "home-services", one namespace per corpus
  sql_store.py       SQLite schema + loader for structured facts (.storage/home_services.db)
  sql_queries.py     prices, quote comparison, quote accuracy, rating trends, trust signals
  entity_resolution.py  links informal mentions ("Mike at Ace") to businesses and experts
  graph_builder.py   builds the Neo4j knowledge graph (nodes/relationships + MENTIONS links)
  graph_queries.py   expansion from vector hits + graph-native queries
  graph_db.py        Neo4j driver (works around missing CA certs on python.org macOS builds)
  retrieval.py       VectorRetriever and GraphRetriever -> a shared EvidenceBundle
  app.py             container holding the three stores (connections cached, SQLite thread-local)
  tools.py           9 agent tools (6 shared, 3 graph-only) + the tool-call recorder
  agent.py           LangGraph agent, system prompt, and answer verification
  embeddings.py      OpenAI text-embedding-3-small
  tool_loadout.py    LangChain middleware that binds only the top-k relevant tools per question
  cli.py             `expert-finder` command
ui/
  Home.py            chat page with evidence and tool-trace panels
  pages/1_Compare.py vector vs graph, side by side
  shared.py          cached resources and rendering helpers
tests/
```

## How the two modes differ

Both use the same model, prompt, Pinecone index and structured SQL tools. Graph mode adds:
1. **expansion of every vector hit** through Neo4j — resolves which business a post refers to,
   pulls in the rest of a discussion thread, attaches quote-vs-invoice to a reviewed job, and
   surfaces businesses the vector search never returned;
2. **three graph-only tools** — neighbourhood word of mouth, which technician people praise, and
   review authenticity (ratings split by whether reviewers are traceable to real customers).

Every answer is checked against what the tools actually returned: citations that match no
retrieved evidence, phone numbers no tool produced, and answers with no citations at all are
all reported as ungrounded.

## Setup

```bash
uv sync
cp .env.example .env                 # fill in keys
uv run expert-finder check           # verify OpenAI, Pinecone, Neo4j, LangSmith
uv run expert-finder index           # embed and upsert into Pinecone (~3K docs, < $0.01)
uv run expert-finder build-sql       # rebuild the SQLite store
uv run expert-finder build-graph --reset   # build the Neo4j graph (~6.1K nodes, ~19.8K relationships)
uv run expert-finder mention-report  # entity-resolution accuracy vs ground truth
uv run expert-finder search "bill was higher than the quote" --namespace reviews \
    --filter '{"category": "hvac", "rating": {"$lte": 2}}'

uv run expert-finder ask "Is Bee Cave Roofing legit?" --mode graph --evidence
uv run streamlit run ui/Home.py     # chat UI + the vector/graph comparison page

uv run pytest                       # integration tests skip without credentials
uv run pytest -m "not integration"  # offline only
```
