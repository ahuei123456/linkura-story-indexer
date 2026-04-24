# Architecture Review: RAG and Vector Retrieval

## Current Architecture

The repository is a lightweight hierarchical-summary RAG system for the Markdown story corpus.

- Story source files live under `story/`.
- `StoryProcessor` reads Markdown files, extracts hierarchy metadata, and splits files into scene-like nodes.
- `HierarchicalSummarizer` creates three summary tiers:
  - Level 3: part summaries
  - Level 2: episode summaries
  - Level 1: year summaries
- `cli.ingest` embeds and upserts only generated summary nodes into Chroma.
- `StoryQueryEngine` embeds the user question, retrieves the top 3 summary records globally, fetches raw Markdown only for level 3 matches, and asks Gemini to synthesize the final answer.
- `StateExtractor` builds `world_state.json` from episode summaries as a separate fact ledger.

The persisted Chroma index currently contains 449 records:

- 400 part summaries
- 45 episode summaries
- 4 year summaries
- 0 raw scene chunks

Raw scenes are parsed in memory, but they are not embedded. Q&A evidence therefore comes from generated summaries, plus a lazy full-file read when a level 3 summary happens to rank in the top 3.

## Planned But Not Implemented

Several intended architecture items are not implemented yet:

- Tier 4 raw scenes are not embedded, even though they should be the source of truth.
- Intent detection for global versus specific questions is not implemented.
- Temporal filtering is not implemented.
- The state ledger is built once from the full corpus, so injected facts can represent end-of-story truth rather than truth at the user's requested point in the story.
- The audit loop for retcons, name consistency, and answer verification is not implemented.
- The agentic/tool-calling RAG loop is not implemented.

These are not necessarily failures for a prototype, but they matter because the implemented system is simpler than the target architecture.

## Core Principle

Use summaries as the routing layer. Use raw scenes as the evidence layer.

Summaries should help find the right area of the corpus. They should not be the final evidence unless the question explicitly asks for a high-level summary.

## Main Criticisms

### Retrieval Is Flat, Not Truly Hierarchical

The stored data has summary tiers, but the query algorithm does not navigate them. `StoryQueryEngine._retrieve` runs a single flat vector search across all tiers with `n_results=3`.

This creates two problems:

- Broad summaries can crowd out precise evidence.
- Part-level raw text is only fetched if a part summary happens to land in the top 3.

The current tiers compete in one pool instead of cooperating in a routing and evidence pipeline.

### Raw Scenes Are Parsed But Not Indexed

`StoryProcessor` creates raw scene nodes in memory, but `cli.ingest` only upserts generated summaries. Raw scene chunks are not embedded into Chroma.

For Q&A, this means the vector database cannot directly retrieve exact dialogue, minor facts, scene details, or localized evidence. The answer generator sees raw Markdown only when a level 3 part summary is retrieved.

This is the largest reliability gap.

### Prompt Claims Raw-Source Grounding More Strongly Than the Context Supports

The query system prompt says the assistant answers strictly from raw source text. However, for level 1 and level 2 matches, `_fetch_raw_text` returns a placeholder saying raw text was omitted and the answer should rely on the summary.

That makes citations weaker than they look. A cited year or episode summary may be a generated intermediate artifact rather than direct source evidence.

Operationally, the system should either soften the prompt to acknowledge summary-backed citations, or preferably gate answer generation on retrieved raw evidence. If no raw evidence is found, it should perform an expansion step or report that it does not have enough source-grounded context.

### Top-K Retrieval Is Too Small

The current `n_results=3` is fragile for:

- multi-hop lore questions
- character-history questions
- "when did this first happen" questions
- questions involving aliases, honorifics, or unit names
- broad thematic questions that need several episodes
- questions where the best evidence is a small detail inside a long part

Three global summary hits are not enough for dependable Q&A over a large narrative corpus.

### No Reranking, Diversity, or Hybrid Search

The retrieval stack has no reranker, no MMR/diversity step, no lexical/BM25 search, no Reciprocal Rank Fusion, and no query expansion.

Dense vector search over English summaries may miss:

- Japanese names used by the user
- English glossary terms not present in a summary
- exact episode or part references
- short proper nouns
- dialogue phrasing
- rare details that were compressed out during summarization

This is especially important because the corpus is bilingual and the summaries are generated in English.

### Metadata Filtering Is Underused

The index stores useful metadata such as `arc_id`, `story_type`, `episode_name`, `part_name`, and `summary_level`, but the query path does not use metadata filters.

There is no routing for questions such as:

- "in 103..."
- "before episode 12..."
- "in the side stories..."
- "as of this point..."
- "what happens in episode 5?"

This also means the system has no protection against future knowledge leaking into answers for temporally scoped questions.

### State Ledger Lacks Provenance and Temporal Semantics

`world_state.json` is extracted from episode summaries, not raw source text. Its facts have no citations, confidence, source episode, source part, source scene, or valid time range.

The merge logic is also lossy:

- character roles are replaced by the longer string
- honorific facts overwrite prior values per target
- `is_active` is OR'd forever
- locations and groups are simple sets

This makes the state ledger useful as a broad glossary/context aid, but weak as authoritative evidence.

### Summary Cache and Vector Index Can Go Stale

The summary cache keys are based on story hierarchy, not source content hashes or pipeline versions.

If a Markdown file, prompt, model, glossary, parser behavior, or schema changes, cached summaries can silently remain stale.

Chroma `upsert` never deletes old records. If a Markdown file is renamed, deleted, or re-split, the old vectors silently remain and can contaminate retrieval while ingestion still appears successful. Stale-vector pruning is a first-class correctness requirement, not a cleanup task.

### Story Ordering Is Hardcoded

`episode_sort_key` hardcodes arc IDs and side-story placement. That works for the current corpus shape, but adding or reordering arcs requires a code change.

Chronological ordering should come from a manifest/config or generated canonical order table, not from hardcoded branching in summarization code.

### Parsing Is Shallow

Scene splitting is a plain `content.split("---")`, and script/prose detection is a simple file-level heuristic. `parse_script_line` exists but the structured speaker/text representation is not used downstream.

That limits future retrieval features such as:

- speaker-specific search
- "who said X?"
- dialogue-only citations
- narrative-vs-dialogue filtering
- character co-occurrence tracking

### Tests Do Not Cover RAG Behavior

The current tests cover package import, scene splitting, script detection, and hierarchy extraction.

They do not cover:

- ingestion end-to-end from file to scene to summary to Chroma upsert
- Chroma metadata shape and required field types
- citation formatting and metadata round-tripping
- deterministic state extraction on fixed inputs
- answer-grounding smoke tests
- cache invalidation behavior
- retrieval quality against known source scenes

The golden-set retrieval eval is one part of this. Unit and integration coverage around the ingestion and retrieval pipeline is the other.

## High-Impact Recommendations

### 1. Index Raw Scene Chunks

Add level 4 raw scene records to Chroma, either in the existing `story_nodes` collection or in a separate `story_scenes` collection.

Each raw scene should include:

- raw text
- normalized retrieval text
- `arc_id`
- `story_type`
- `episode_name`
- `part_name`
- `scene_index`
- canonical story order
- parent part/episode/year IDs
- file path
- detected speakers
- `is_prose`

This should become the primary evidence layer for Q&A.

### 2. Use Summaries for Routing, Raw Scenes for Answers

A better query flow would be:

1. Retrieve broad candidates from summaries.
2. Use those candidates to identify likely arcs, episodes, and parts.
3. Expand into raw scene chunks and neighboring scenes.
4. Rerank raw chunks.
5. Generate the answer from raw chunks.
6. Cite only raw chunks or source-backed part labels.

Summaries should help find the right area of the corpus. They should not be the final evidence unless the question is explicitly asking for a high-level summary.

### 3. Gate Answers on Raw Evidence

Until raw scenes are indexed, the current prompt should be softened because the system sometimes answers from summaries. Once raw scenes are indexed, answer generation should require at least one raw evidence chunk for factual Q&A.

If retrieval only finds broad summaries, the engine should:

1. Expand from summary hits to child parts/scenes.
2. Retry retrieval against raw scenes.
3. Ask the answer model to state insufficient source context if raw evidence still cannot be found.

This directly addresses the prompt-vs-reality mismatch.

### 4. Increase Candidate Recall Before Narrowing

Instead of retrieving only 3 records, retrieve a larger candidate set:

- top 20-50 summary records for routing
- top 30-100 raw scene records for direct evidence
- final 5-12 chunks after reranking

These values are starting points, not fixed constants. Candidate counts should be tuned empirically against the retrieval evaluation set.

The final prompt can stay small, but the retrieval stage needs higher recall.

### 5. Add Neighbor-Scene Expansion

After a raw scene hit, also pull adjacent scenes from the same part.

Narrative evidence often spans scene boundaries:

- one scene establishes the setup
- the next scene contains the key dialogue
- the following scene resolves or clarifies the event

Neighbor expansion should be bounded, such as previous/next one or two scenes, and then reranked with the original candidates.

### 6. Add Hierarchical Routing With RRF

Use the tier structure intentionally:

1. Search summaries and raw scenes separately.
2. Search per tier or per intent when appropriate.
3. Fuse candidates with Reciprocal Rank Fusion.
4. Fan out from level 1 or level 2 hits to child episodes, parts, and scenes.
5. Rerank the expanded raw evidence candidates.

RRF is a good fit because it combines multiple ranking sources without requiring score calibration.

### 7. Add Hybrid Retrieval

Combine dense vector search with lexical search.

Dense search is good for semantic similarity, but lexical search is better for:

- exact names
- episode numbers
- Japanese terms
- quoted dialogue
- glossary terms
- rare proper nouns

For this project, even a simple local BM25, SQLite FTS, Whoosh, or Tantivy side index would be a major improvement.

### 8. Add Glossary-Based Query Expansion

Before dense retrieval, expand the user query with aliases from `glossary.json`.

Example:

```text
Kaho -> Kaho Hinoshita / 花帆 / 日野下花帆
```

This is cheap and high impact because the source corpus is Japanese, summaries are English, and users may ask with either Japanese or English names.

### 9. Add Reranking

After initial retrieval, rerank candidates using a cross-encoder, an LLM scoring pass, or structured heuristics.

Useful reranking signals:

- exact character/name matches
- glossary alias matches
- episode/arc constraints
- whether the chunk contains dialogue from mentioned speakers
- proximity to other high-scoring chunks
- summary parent score

A practical first version could score top candidate chunks with a Gemini Flash call before passing the final top-k to the answer generator. The number of reranked candidates should be tuned with the eval set.

### 10. Implement Query Routing and Metadata Filters

Add a lightweight query analysis step that extracts:

- arc/year constraints
- episode constraints
- side story vs main story
- character names and aliases
- temporal phrases
- whether the user wants summary, exact evidence, comparison, chronology, or a quantitative answer

Use this to apply Chroma metadata filters and to choose whether to search summaries, raw scenes, or both.

Do not rely on string comparison such as `arc_id <= target` for temporal filtering. Add canonical numeric order fields instead.

### 11. Add Quick Metadata Filters Before Full Routing

Before the full query router exists, add simple Chroma `where` filters where they are obvious:

- `summary_level` for high-level summary questions versus detail questions
- `arc_id` when the user explicitly names an arc
- `story_type` for main-story versus side-story questions

This is not a substitute for proper query analysis, but it is a cheap way to reduce obvious retrieval noise.

### 12. Add Temporal Ordering

Create canonical story-order fields during ingestion. These should be independent of filesystem traversal order and should account for the special ordering of main stories and side stories.

Story order should be driven by a manifest/config or canonical order table so adding a new arc does not require editing `episode_sort_key`.

Then support filters such as:

- before a given arc/episode/part
- after a given event
- as of a given episode
- main story only
- side story only

This matters for narrative QA because "what is true" changes over time.

### 13. Version the Index and Cache

Add an ingestion manifest containing:

- source file hashes
- parser version
- summarization prompt version
- glossary hash
- model names
- embedding model name
- schema version
- ingestion timestamp

Use this to invalidate stale cache entries and delete stale Chroma records. Stale-vector pruning should handle renamed files, deleted files, and changed scene boundaries.

### 14. Give State Facts Provenance

Replace or supplement the current world-state structure with source-backed fact records:

- subject
- predicate
- object/value
- arc/episode/part/scene source
- valid-from story position
- valid-to story position, if superseded
- confidence
- extracted quote or source pointer

State facts should be extracted from raw scenes, not generated summaries. The ledger should support routing, consistency checks, and glossary-like context, but final answers should still cite source chunks.

### 15. Preserve Structured Dialogue and Narrative Beats

Use `parse_script_line` output to store structured dialogue turns.

For prose files, store quoted dialogue separately from narrative beats so both can be queried independently.

That enables:

- speaker-specific retrieval
- "who said X?" queries
- better co-occurrence extraction
- cleaner citations
- character voice and honorific analysis
- code-based counting for quantitative questions

### 16. Build a Retrieval Evaluation Set

Create a small gold dataset of Q&A cases with expected source files or scenes.

Track:

- recall@k for expected scenes
- reranker hit rate
- answer faithfulness
- citation correctness
- temporal leakage
- glossary consistency

This project will be hard to improve safely without retrieval tests. Candidate counts, rerank cutoffs, neighbor-scene expansion width, and hybrid weighting should be tuned against this dataset rather than guessed.

### 17. Expand Pipeline Test Coverage

Add focused tests around the pipeline, not just parser helpers:

- ingestion end-to-end from Markdown file to scene nodes to Chroma upsert
- Chroma metadata shape, including required fields and types
- citation formatting from metadata, with round-trip parse tests if labels become structured
- state extraction on fixed deterministic inputs
- answer-grounding smoke tests that verify retrieved chunks contain claimed evidence
- cache invalidation behavior when source, prompt version, glossary hash, or schema version changes

These tests should complement the golden retrieval eval. The eval measures search quality; these tests catch pipeline regressions.

## Quick Wins

These are smaller implementation changes that can improve the current system without redesigning the whole pipeline.

### Set Gemini Embedding Task Types

The installed `google-genai` package supports `EmbedContentConfig.task_type`.

Use:

- `RETRIEVAL_DOCUMENT` when embedding documents at ingest time
- `RETRIEVAL_QUERY` when embedding user questions at query time

This should improve dense retrieval quality at low cost.

### Batch Embedding Calls

`database.embed_texts` currently loops and calls `embed_content` once per text. The Gemini SDK accepts a list of contents, so ingestion should batch embeddings where possible.

This reduces network overhead and makes ingestion faster.

### Cache Expensive Clients

Cache these as module-level singletons or explicit application objects:

- `chromadb.PersistentClient`
- Chroma collection handle
- `genai.Client`
- `GoogleModel`

The current code rebuilds clients repeatedly.

### Make Model Names Configurable

`CHAT_MODEL` and `EMBEDDING_MODEL` are hardcoded preview model names in `database.py`.

Load them from environment variables with sensible defaults, such as:

- `LINKURA_CHAT_MODEL`
- `LINKURA_EMBEDDING_MODEL`
- `LINKURA_CHROMA_DB_PATH`

This makes model upgrades and deployment differences easier to manage.

### Improve State Ledger Injection

The current engine injects state ledger entries for arcs that happened to appear in retrieval. That is better than injecting every arc, but still not ideal.

Instead:

1. Determine arc relevance from the question and routing stage.
2. Inject only relevant ledger slices.
3. Use compact JSON instead of `indent=2` to reduce prompt tokens.
4. Prefer source-backed facts once fact provenance exists.

### Improve Raw Fetching and Citations

Once raw scene records are indexed, fetch by `(file_path, scene_index)` rather than reading whole part files.

Citation labels should remain readable, but the context metadata should include enough detail to verify evidence:

- file path
- scene index
- part name
- episode label
- story order

Avoid exposing noisy file paths in every final answer unless the user asks for detailed sources.

## Lower-Priority or Roadmap Items

These items compound on retrieval quality. Running them over a weak index can amplify bad retrieval, so they should come after raw-chunk indexing, hybrid search, reranking, and evaluations are solid.

### Agentic Query Loop

A tool-calling agent could help with multi-hop questions, especially because `pydantic-ai` is already in the dependency set.

Possible tools:

- `search(query, tier?, arc_id?)`
- `get_scene(path, scene_index)`
- `get_state(arc_id, as_of_episode?)`
- `lookup_glossary(term)`

This is useful, but it should come after a strong non-agentic retrieval pipeline with raw chunks, hybrid search, reranking, and evaluations.

### Audit Pass

A secondary LLM call could compare the draft answer against retrieved sources, the state ledger, and the glossary to flag:

- unsupported claims
- wrong names
- glossary violations
- temporal inconsistencies
- honorific mistakes

This is valuable, but it should not substitute for better retrieval and source-grounded citations.

### Quantitative Query Support

For questions like "how many times does Kaho appear as a speaker in Year 103?", use structured parsed dialogue and code-based counting rather than asking the LLM to infer counts from retrieved text.

This requires preserving speaker metadata first.

### Chat Streaming

Streaming answers in `chat` would improve interactivity, but it is not a retrieval-quality issue and should be lower priority.

### Cache Storage Format

Moving `summaries_cache.json` to SQLite or JSONL may help if the corpus grows. For now, content-hash invalidation and stale-vector pruning are more important than the storage format.

Format migration alone fixes nothing if stale cache entries and stale Chroma vectors are still accepted as valid.

### `_node_id` Cleanup

`_node_id` includes `scene_index`, but summary nodes set it to `-1`, so it does not disambiguate summary records. This is not a serious bug because level, episode, and part already disambiguate summaries. It only becomes more important when raw scene records are added.

## Suggested Target Architecture

```text
Markdown story files
        |
        v
Parser / normalizer
        |
        +--> raw scene/dialogue records
        |         |
        |         +--> vector index
        |         +--> lexical index
        |
        +--> part summaries
        +--> episode summaries
        +--> year summaries
        |         |
        |         +--> vector index
        |         +--> lexical index
        |
        v
Index manifest + cache + stale-vector pruning
        |
        v
Query analysis
        |
        v
Metadata filters + glossary expansion
        |
        v
Summary routing + raw scene retrieval + hybrid search
        |
        v
RRF / reranking / neighbor expansion
        |
        v
Answer generation from raw evidence + source citations
        |
        v
Optional audit pass
```

## Priority Order

1. Index raw scene chunks and retrieve them for Q&A.
2. Gate factual answers on raw evidence, or soften the prompt until raw evidence is available.
3. Set Gemini embedding task types and batch embedding calls.
4. Add glossary-expanded hybrid retrieval for proper nouns and bilingual queries.
5. Increase candidate count, add neighbor-scene expansion, and rerank before generation.
6. Add hierarchical routing with RRF and child expansion.
7. Add quick metadata filters, then full query analysis and canonical temporal story-order fields.
8. Move story ordering to a manifest/config.
9. Add cache/index versioning and stale-vector pruning.
10. Add provenance to state facts, extracted from raw scenes.
11. Preserve structured dialogue, narrative beats, and speaker metadata.
12. Add retrieval evals and pipeline tests.
13. Consider agentic loop and audit pass only after the base retrieval pipeline is solid.

## Bottom Line

The current system is a good prototype for browsing generated summaries of a large story corpus. It is not yet a dependable source-grounded Q&A RAG system.

The highest-impact architectural change is to make summaries a routing layer and raw scene chunks the evidence layer. The highest-impact quick wins are embedding task types, batched embeddings, glossary query expansion, prompt/evidence gating, and stale-index hygiene.

Once raw source chunks are indexed, reranked, and cited, the rest of the architecture can become much more reliable without requiring a major rewrite.
