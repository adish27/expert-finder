# Home Services Expert Finder: Application Overview

Home Services Expert Finder is a conversational AI assistant that helps homeowners in the Austin, TX
metro find the right professional for a home project. It covers plumbing, electrical, HVAC, roofing and
landscaping. Users ask in their own words, often vaguely: "my AC is blowing warm air and it's 104 out,"
"is $14,000 a fair price for a new roof in Round Rock?", "who do people in Hyde Park use for sprinklers?"
The assistant works out what they need and replies like a well-informed neighbor. It names specific
businesses and the experts who work there, says what the job should cost, and explains why, citing the
reviews, neighborhood posts and real quotes behind each claim. It holds a multi-turn conversation, asks a
follow-up question when the request is ambiguous (where do you live? repair or replace?), and remembers
the homeowner's context across sessions: their city, their home, past projects.

Behind the conversation, an agent built with LangGraph decides which data sources and tools each question
needs, instead of searching one big index:
- **Semantic search** (Pinecone with OpenAI embeddings) over customer reviews from six platforms, social
  posts from Nextdoor, Facebook, Reddit, TikTok and Instagram, and business profiles.
- **Structured tools** for facts search can't answer reliably: service prices and price ranges, comparing
  estimates, service areas, emergency availability and license status.
- **A knowledge graph** (Neo4j) that links businesses, experts, services, customers, neighborhoods, reviews
  and posts. It answers relationship questions such as which technician people praise, what neighbors
  recommend, or whether a business has gotten worse lately.

Answers must be grounded. Every recommendation cites its evidence, and the assistant says so when the
evidence is thin rather than guessing. It also surfaces red flags: invoices well above the quote,
unresolved complaints, expired licenses and suspicious bursts of reviews. mem0 holds user memory, and
LangSmith traces every step of retrieval and reasoning.

The application is also an experiment comparing two retrieval approaches on the same questions:
- **Vector RAG** answers from semantically similar text chunks alone.
- **Graph RAG** starts from those chunks, then traverses Neo4j to pull in connected entities and facts.

Both use the same model, prompts and tools, so differences in the answers come from retrieval. A comparison
view shows the two answers side by side with the evidence each one used. An evaluation harness scores them
against the dataset's hidden ground truth:
- whether the recommended businesses are actually good,
- whether trust problems are detected,
- whether informal mentions like "Mike at the plumbing place" are resolved to the right person,
- citation accuracy, latency and cost.

**Example questions the assistant should handle**
- "Water heater is leaking in the garage. Who can come out today in Cedar Park?"
- "I got three quotes for a new AC: $9,800, $12,600 and $15,900. Which should I take?"
- "Who's the best electrician for a Tesla charger install, and is there a specific tech people ask for?"
- "Is Bee Cave Roofing legit? I'm seeing a lot of 5-star reviews."
- "What do my neighbors in Mueller say about landscapers for a front-yard xeriscape?"
- "Has Hill Country Air gotten worse? A friend had a bad experience recently."

**Scope.** Version 1 uses the synthetic dataset in `data/home_services/`: 100 businesses, 5 categories, the
Austin metro. It recommends and explains. It does not book appointments, take payments or contact
businesses. More categories (remodeling, painting, general contractors, handyman) are planned for later.
