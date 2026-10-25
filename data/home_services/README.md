# Home Services Expert Finder — Dataset (v1, synthetic)

Phase 1 of the home-projects expert finder: a structured, multi-source dataset of local
home-service businesses, the experts who work there, what they offer and charge, real
quotes, customer reviews from several review platforms, and social chatter.

**Everything is fictional.** Cities, neighborhoods, platform names, product brands and
licensing bodies are real (for realism); businesses, people, reviews and posts are generated.
License numbers contain `SYN`, phones use the reserved `555-01xx` range, websites use `.example`.

Regenerate (deterministic, seed `20261025`):
```bash
uv run python scripts/generate_home_services_data.py
```
Sources: [scripts/home_services_catalog.py](../../scripts/home_services_catalog.py) (domain knowledge:
services, market prices, geography, platforms) and
[scripts/generate_home_services_data.py](../../scripts/generate_home_services_data.py) (sampling + validation).

---

## 1. Use-case plan

**User:** a homeowner in the Austin TX metro planning a repair or project.
**Goal:** find the right business and the right person, at a fair price, with evidence from what other customers say.

Questions the system should eventually answer (this dataset is designed to support each):

| Question type | Example | Data needed |
|---|---|---|
| Find by need | "AC blowing warm air in Round Rock, who can come today?" | services, service area, emergency hours, response time |
| Price sanity check | "Is $14,200 for a new 4-ton AC fair in Pflugerville?" | offerings price ranges, estimates, social price checks |
| Compare quotes | "Compare my three roof quotes" | estimates grouped by `customer_project_id`, line items |
| Trust and quality | "Which plumbers bill exactly what they quote?" | estimates vs final invoice, review text, sub-ratings |
| Specific expert | "Who is the best tech at X for tankless installs?" | experts, specialties, review/social mentions of first names |
| Neighborhood word of mouth | "Who do my Hyde Park neighbors recommend?" | Nextdoor/Facebook threads, customer neighborhoods |
| Red flags | "Any complaints about X?", "Are these reviews real?" | BBB complaints, Reddit PSAs, review bursts, license status |
| Trend | "Has X gotten worse lately?" | review timestamps (the `declining` profiles drop after May 2025) |

**Phases**
1. **Dataset (this folder)**: structured records + unstructured text + ground truth.
2. **Ingestion & graph**: load into Neo4j; chunk/embed review and post text into Chroma; resolve informal social mentions to businesses/experts.
3. **Retrieval**: agentic routing across corpora (reviews, social, offerings) + graph traversal (business → service → price, customer → neighborhood → recommendations).
4. **Answering & evaluation**: grounded answers with citations, scored against `ground_truth/`.

Categories in v1: **plumbing, electrical, HVAC, roofing (incl. gutters), landscaping (front yard,
backyard, irrigation, trees, hardscape)**. General contractors, kitchen/bath remodeling, painting and
handyman are reserved in `reference/categories.json` (`in_dataset: false`) for a later phase.

## 2. What's in it

| File | Rows | Grain |
|---|---:|---|
| `businesses.jsonl` | 100 | one business (20 per category) |
| `experts.jsonl` | 325 | one person at a business (owner, techs, estimators, designers) |
| `service_offerings.jsonl` | 999 | one service a business offers, with its price range |
| `customers.jsonl` | 1,400 | one homeowner, with home details and per-platform handles |
| `estimates.jsonl` | 1,258 | one written quote with line items; accepted ones include the job outcome |
| `reviews.jsonl` | 2,041 | one review/complaint on Google, Yelp, Angi, Thumbtack, BBB or Houzz |
| `social_posts.jsonl` | 884 | one post or reply on Nextdoor, Facebook groups, Reddit, TikTok or Instagram |
| `reference/categories.json` | 10 | category taxonomy, licensing body |
| `reference/services.json` | 66 | service catalog: pricing model, unit, market price range, aliases, seasonality |
| `reference/service_areas.json` | 10 cities | cities → neighborhoods → zip, typical home age |
| `reference/platforms.json` | 11 | review/social platform capabilities (titles, cost field, sub-ratings, verified hire) |
| `ground_truth/*` | — | hidden labels for evaluation (see §5); **not** for ingestion |
| `manifest.json` | — | counts, seed, date range, disclaimer |

Format: **JSON Lines** for record sets (one JSON object per line; nested fields like line items and
sub-ratings stay intact; `pd.read_json(path, lines=True)` works), **JSON** for small reference tables.
Conventions: ids are opaque prefixed strings (`biz_0001`, `exp_0001`, `off_00001`, `cust_00001`,
`est_00001`, `rev_000001`, `post_00001`, `thr_0001`, `proj_00001`); timestamps are ISO 8601 with the
America/Chicago offset; dates are `YYYY-MM-DD`; money is USD in fields ending `_usd`; missing values are `null`.
Date range: 2021-01-01 to 2026-10-25.

### Entity relationships

```
reference/services ─┐
                    ├─< service_offerings >── businesses ──< experts
categories ─────────┘          │                  │  └── owner_expert_id → experts
                               │                  │
customers ──< estimates >──────┘ (business, expert, service; customer_project_id groups competing quotes)
    │             └── job (accepted): final invoice, change orders
    ├──< reviews >── businesses   (estimate_id when the reviewed job had a written quote)
    └──< social_posts             (thread_id / parent_post_id form threads; mentions are free text)
```

## 3. Data dictionary

### businesses.jsonl
| Field | Type | Notes |
|---|---|---|
| `business_id` | str | `biz_0001` |
| `name`, `legal_name` | str | DBA name and registered name |
| `primary_category`, `secondary_categories` | str, [str] | category ids; 18 businesses also do an adjacent trade (e.g. plumbing + HVAC) |
| `description` | str | short profile blurb |
| `founded_year`, `business_size`, `employee_count` | int, enum, int | size: `solo_operator`, `small_team`, `established`, `regional` |
| `owner_expert_id` | str | → experts |
| `contact` | obj | `phone`, `email`, `website` |
| `address` | obj | `street`, `city`, `state`, `zip` |
| `service_area` | obj | `cities` [str], `radius_miles` |
| `hours` | obj | `mon_fri`, `sat`, `sun` (`null` = closed) |
| `emergency_service` | obj | `available`, `hours`, `after_hours_surcharge_pct` |
| `licenses` | [obj] | `type`, `issuer`, `number`, `status` (`active`/`expired`), `expires` |
| `insurance` | obj | `general_liability`, `liability_coverage_usd`, `workers_comp`, `bonded` |
| `certifications`, `brands`, `languages`, `payment_methods` | [str] | |
| `financing_available` | bool | |
| `warranty` | obj | `workmanship_months`, `notes` |
| `pricing_policy` | obj | `hourly_rate_usd`, `service_call_fee_usd`, `service_call_fee_waived_with_repair`, `free_estimates`, `minimum_charge_usd`, `discounts` |
| `response_time` | obj | `typical_reply_hours`, `typical_days_to_schedule` |
| `social_accounts` | obj | any of `facebook`, `instagram`, `tiktok`, `nextdoor` |
| `listed_on` | [str] | review platforms where the business has reviews |

### experts.jsonl
`expert_id`, `business_id`, `first_name`, `last_name`, `goes_by` (nickname customers use, e.g. Mike),
`role`, `is_owner`, `field_role` (false for office staff), `years_experience`, `licenses` [{type, issuer, number}],
`certifications`, `languages`, `specialties` [service_id], `bio`.

### service_offerings.jsonl
`offering_id`, `business_id`, `service_id`, `category`, `is_secondary_category`, `pricing_model`
(`flat`, `hourly`, `per_unit`, `per_sqft`, `per_linear_ft`, `per_square` (=100 sq ft of roof), `per_zone`,
`per_visit`, `per_tree`, `per_year`), `unit`, `price_low_usd`, `price_high_usd`, `typical_price_usd`
(per unit), `price_is_starting_at`, `includes` [str], `excludes` [str], `permit_required`, `lead_time_days`,
`warranty_months`, `available_for_emergency`, `notes`.

### customers.jsonl
`customer_id`, `first_name`, `last_name`, `city`, `neighborhood`, `zip`,
`home` {`type`, `year_built`, `sq_ft`, `stories`, `has_pool`},
`platform_handles` {google, yelp, angi, thumbtack, houzz, bbb, nextdoor, facebook, reddit, tiktok, instagram} —
how this person appears on each platform (`null` = no account). Lets you link one person's
review and Nextdoor reply about the same job.

### estimates.jsonl
| Field | Notes |
|---|---|
| `estimate_id`, `customer_project_id` | estimates sharing a project id are competing quotes for one job |
| `business_id`, `expert_id`, `customer_id`, `service_id`, `category` | |
| `request_channel` | phone, website form, angi, thumbtack, referral, nextdoor, google |
| `requested_at`, `site_visit_at`, `issued_at`, `valid_until` | |
| `job_address` | city, neighborhood, zip |
| `job_description` | customer's problem + proposed scope (text) |
| `quantity`, `unit` | e.g. 28 squares, 6 zones, 1 job |
| `line_items` | [{description, category (labor, materials, permit, disposal, discount), quantity, unit, unit_price_usd, total_usd}] |
| `subtotal_usd`, `sales_tax_usd`, `total_usd` | tax = 8.25% of materials |
| `financing_offered` | |
| `status`, `decline_reason` | accepted, declined, expired, pending |
| `job` | for accepted: `job_id`, `completed_at`, `final_invoice_usd`, `change_orders` [{description, amount_usd}], `duration_days` |

### reviews.jsonl
| Field | Notes |
|---|---|
| `review_id`, `platform` | google, yelp, angi, thumbtack, bbb, houzz |
| `business_id`, `customer_id` | `customer_id` is `null` for reviews not traceable to a customer |
| `reviewer_display_name`, `reviewer_total_reviews` | platform-style name; account history (1-2 = brand-new account) |
| `review_type` | `review` or `complaint` (BBB formal complaint) |
| `rating` | 1-5; `null` for complaints |
| `title` | Angi and Houzz only |
| `text` | free text; style differs by platform (Google short, Yelp long, BBB formal); some Google ratings have empty text |
| `created_at`, `job_date`, `service_id`, `estimate_id` | |
| `cost_reported_usd` | Angi/Thumbtack/Houzz cost field |
| `verified_hire` | Angi/Thumbtack |
| `sub_ratings` | Angi: quality, value, punctuality, professionalism, responsiveness |
| `photos_count`, `helpful_count` | |
| `complaint_status` | BBB: resolved, unresolved, answered |
| `owner_response` | {text, created_at} or `null` |

### social_posts.jsonl
| Field | Notes |
|---|---|
| `post_id`, `platform` | nextdoor, facebook, reddit, tiktok, instagram |
| `post_type` | recommendation_request, reply_recommendation, reply_warning, reply_other, price_check, reply_price_comparison, warning (PSA), project_showcase, price_reveal, homeowner_recommendation, storytime_complaint |
| `thread_id`, `parent_post_id` | threads on Nextdoor/Facebook/Reddit; `null` for standalone posts |
| `community` | "Hyde Park, Austin" (Nextdoor), group name (Facebook), subreddit |
| `author` | {handle, display_name, account_type: homeowner or business} |
| `customer_id` | set when the author is a known customer |
| `created_at`, `text`, `hashtags` | |
| `media` | [{type: video, duration_s, transcript} or {type: photo, alt_text}] |
| `engagement` | likes, comments, shares, views (platform-dependent; `null` when not applicable) |

Businesses are mentioned **informally** in posts: full name, short name, lowercase, "Mike at Ace Plumbing",
"Carlos the plumber", or an @handle. Resolving these mentions is part of the problem; the answers are in `ground_truth/social_mentions.jsonl`.

## 4. How the data hangs together

Each business has a hidden **profile** (`ground_truth/business_profiles.jsonl`): an archetype, a price tier
and traits for quality, punctuality, communication and upselling. Every job outcome is sampled from that profile, and the
outcome drives the rating, the review sentences, invoice overruns, owner responses and social posts. The
sources therefore agree with each other, and the patterns can be recovered from the observable data:

| Archetype (per category) | n | Observable signature |
|---|---:|---|
| reliable_veteran | 5 | many reviews, ~4.4★, invoices match quotes |
| premium_craftsman | 2 | highest prices, ~4.8★, longer scheduling lead times |
| budget_value | 4 | lowest prices, ~3.9★, punctuality and communication complaints |
| new_and_eager | 2 | founded 2023-25, few reviews, very responsive |
| inconsistent | 3 | polarized ~2.8★, frequent overruns, defensive owner responses |
| declining | 2 | ~4.6★ before May 2025, ~2.6★ after |
| upsell_heavy | 1 | premium prices, "pushed a new system" reviews, ~40% of jobs over quote |
| suspicious_reviews | 1 | burst of generic 5★ reviews from new accounts; expired license; real experience poor |

Other realism features:
- **Seasonality:** AC failures in summer, furnace repairs in winter, spring hail and roof claims, and a burst-pipe spike during the Feb 2021 freeze.
- **Home age by neighborhood:** 1950s Hyde Park bungalows versus newer suburban builds.
- **Prices:** quotes stay within each business's offering range and scale with its price tier.
- **Comparison shopping:** big jobs get 1–3 competing quotes under one `customer_project_id`, with decline reasons.
- **Cross-platform overlap:** about 7% of customers review the same job on two platforms. Social replies come from real customers describing their actual jobs, with dates after the job.
- **Rating bias:** platform choice depends on the rating. BBB skews to complaints, and Houzz to landscaping and roofing.

## 5. Ground truth (for evaluation only)

| File | Use |
|---|---|
| `business_profiles.jsonl` | archetype, tier, traits, decline date: score "who is reliable / overpriced / getting worse" |
| `jobs.jsonl` | every job's outcome flags (quality_ok, on_time, communicated, upsold, overrun_pct) |
| `review_flags.jsonl` | `suspicious_burst` and `cross_platform_duplicate` labels |
| `review_expert_mentions.jsonl` | which expert a review's first-name mention refers to |
| `social_mentions.jsonl` | gold business/expert links and sentiment for every mention in posts |

Keep these out of the index and prompts. They exist to measure retrieval and answer quality.

## 6. Known simplifications
- Review and post text is template-composed. It is varied and coherent with outcomes, but less diverse than real writing. An LLM paraphrase pass is a possible v2 step.
- Single metro (Austin). Prices are rough 2026 market ranges, not quotes from real companies.
- Customers are only the people who appear in estimates, reviews or posts. No full CRM.
- Sales tax on estimates is simplified to 8.25% of the materials line.
