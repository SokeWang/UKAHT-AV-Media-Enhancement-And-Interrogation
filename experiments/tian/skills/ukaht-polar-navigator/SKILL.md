---
name: ukaht-polar-navigator
description: Expert archival navigation, entity normalization, and evidence-grounded search across UKAHT Antarctic heritage collections (Base E Stonington, Base W Detaille, Base A Port Lockroy). Use when users ask to search, filter, or analyze historical polar photographs, stations, expeditions, or conservation timelines.
---

# UKAHT Polar Navigator Skill

This skill implements the **Progressive Disclosure Architecture** for exploring the 1,584 UKAHT historical photographic archive.
Rather than dumping massive schemas into every turn, context and validation rules are progressively activated across four distinct phases:

```
┌─────────────────────────────────────────────────────────────┐
│ Level 1: Intent Discovery (Triggered by user query)         │
├─────────────────────────────────────────────────────────────┤
│ Level 2: Entity & Era Normalization (Heuristic resolution)   │
├─────────────────────────────────────────────────────────────┤
│ Level 3: Tool Execution & Dynamic Fallback (Zero-result fix)│
├─────────────────────────────────────────────────────────────┤
│ Level 4: Grounded Synthesis & Citation Validation           │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Progressive Entity Normalization

When user mentions colloquial place names, dates, or photographer credits, resolve them against the domain ontology (`references/polar_heritage_ontology.json`):

1. **Station Resolution**:
   - `Stonington Island` / `Station E` -> `base_code = 'E'`
   - `Detaille Island` / `Station W` -> `base_code = 'W'`
   - `Port Lockroy` / `Goudier Island` -> `base_code = 'A'`

2. **Temporal & Decade Resolution**:
   - `1950s` / `50s` -> `shooting_year BETWEEN '1950' AND '1959'` or `shooting_year='1950s'`
   - `before 1970` -> `shooting_year < '1970'`
   - `after 2000` / `modern` -> `shooting_year > '2000'`
   - `2021-22` -> `shooting_year = '2021-22'`

3. **Domain Categories**:
   - Map keywords to standard UKAHT clusters: `'Polar Landscape & Glaciers'`, `'Artifacts & Museum Display'`, `'Exterior Heritage & Huts'`, `'Expedition Equipment & Vessels'`.

---

## Phase 2: Adaptive Tool Routing & Execution

Route the normalized intent to the most specific tool:

- **Visual/Atmospheric Queries** (e.g. *"seals on ice floes"*, *"weathered timber"*):
  -> Call `semantic_search(query=...)`.
- **Structured Metadata Queries** (e.g. *"Base E before 1970"*, *"photos by Mike Cousins"*):
  -> Call `sql_filter(base_code=..., shooting_year=..., category=...)`.
- **Complex Text-to-SQL Queries** (e.g. multi-year comparisons, compound ordering):
  -> Call `sql_query(where_clause=...)`.

---

## Phase 3: Dynamic Fallback Recovery Protocol

If `sql_filter` returns **0 results** (due to strict metadata mismatch):
1. Intercept the zero-result response.
2. Automatically trigger fallback to `semantic_search(query=keyword)`.
3. Preserve search continuity without surfacing an empty dead-end to the user.

---

## Phase 4: Fact-Grounding & Evidence Citation

Every historical claim must be verifiable:
- Explicitly cite photograph IDs using `[ID: ukaht_xxxxxxxxx]`.
- For extended timeframe queries (e.g. "how Base E changed over 100 years"), clarify the exact recorded years in the archive (1944–2025) and anchor observations to cited assets.
