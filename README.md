# Home Services Expert Finder

Helps homeowners find the right plumber, electrician, HVAC tech, roofer or landscaper, at a fair
price, backed by evidence from reviews and neighbors. Planned as a Graph RAG application with
Chroma for vector search, Neo4j for relationships, and LangChain 1.x + OpenAI for agents and answers.

**Status:** phase 1 (synthetic dataset) is done. The retrieval and agent layers are next.

## Layout

```
data/home_services/        synthetic dataset (JSONL + reference JSON); see its README for the plan and data dictionary
scripts/
  home_services_catalog.py        domain knowledge: services, market prices, geography, platforms
  generate_home_services_data.py  deterministic generator + validation
src/expert_finder/
  config.py         settings from env / .env (models, Neo4j, storage, dataset path)
  embeddings.py     OpenAI text-embedding-3-small, or an offline hashing fallback
  tool_loadout.py   LangChain middleware that binds only the top-k relevant tools per question
tests/test_dataset.py   dataset integrity checks
docker-compose.yml      local Neo4j
```

## Setup

```bash
uv sync
cp .env.example .env                 # add OPENAI_API_KEY
docker compose up -d neo4j           # optional until the graph layer lands
uv run python scripts/generate_home_services_data.py   # regenerate the dataset (deterministic)
uv run pytest
```
