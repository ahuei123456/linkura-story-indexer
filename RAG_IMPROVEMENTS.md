# RAG Retrieval Improvements (v4)

Fourth-pass notes. Supersedes `RAG_IMPROVEMENTS_3.md` by folding in the
remaining useful items from `ARCHITECTURE_REVIEW_3.md`: the operational
fallback flow for missing raw evidence, starting-point candidate counts,
the full set of named tunable knobs, bounded neighbor expansion,
story-ordering promoted to a first-class recommendation, and the
`_node_id` conditional escalation. Structure switched from time buckets
(hours / days / bigger) to **scope buckets** (quick wins / pipeline
redesign / data-layer / roadmap) because what actually matters when
planning is "local edit vs. redesign," not wall-clock estimate.

Pairs with `PLAN.md` — items marked (PLAN) are things `PLAN.md` promises
but aren't implemented.

## Index composition today

As of the last ingest the Chroma index holds **449 records**:

- 400 part summaries (Tier 3)
- 45 episode summaries (Tier 2)
- 4 year summaries (Tier 1)
- 0 raw scene chunks (Tier 4)

Raw scenes are parsed in memory by `StoryProcessor` but never embedded.
Q&A evidence therefore comes entirely from generated summaries, plus a
lazy full-file read when a Tier-3 summary happens to rank in top-3.

## Known gaps vs. PLAN.md

- Tier 4 (raw scenes) is not embedded — biggest reliability gap.
- Intent detection (Global vs. Specific) — not implemented.
- Temporal filtering ("don't let the LLM know the future") — not
  implemented. State Ledger is built once from the full corpus, so
  injected "current truth" is always end-of-story truth.
- Agentic / tool-calling RAG (Phase 6) — not implemented.
- Audit loop for retcons / name-consistency (Phase 5) — not implemented.

## Core principle

**Summaries are the routing layer. Raw scenes are the evidence layer.**
Summaries help find the right area of the corpus; they should not be the
final evidence unless the question explicitly asks for a high-level
summary. State facts belong with raw scenes too — extract from scenes,
not from generated summaries, so provenance is preserved.

## Prompt-vs-reality mismatch (sharp bug)

The query system prompt claims the assistant answers "based strictly on
provided raw source text." But `_fetch_raw_text` only returns actual
source text for Tier-3 hits; for Tier-1 and Tier-2 matches it returns a
placeholder telling the LLM to rely on the summary. Citations labelled
as evidence can silently be generated summaries, not source lines.

The fix is a retrieval **policy**, not just a prompt tweak. When
retrieval finds only broad summaries:

1. Expand from summary hits to child parts/scenes.
2. Retry retrieval against raw scenes (with reranking).
3. If still no raw evidence, have the answer model explicitly report
   insufficient source context rather than paraphrasing the summary.

Until Tier 4 is indexed, the system prompt should be softened to
acknowledge summary-backed citations. Once Tier 4 is indexed, factual
Q&A should be gated on at least one raw evidence chunk.

## Quick wins (local edits, no pipeline redesign)

1. **Set `EmbedContentConfig.task_type`** on Gemini embeddings:
   `RETRIEVAL_DOCUMENT` at ingest, `RETRIEVAL_QUERY` at query time.
2. **Batch `embed_content` calls** in `database.embed_texts` — the SDK
   accepts a list of contents; current code loops one-per-text.
3. **Metadata filter the existing retrieval.** Pass
   `where={"summary_level": ...}` / `where={"arc_id": ...}` to Chroma
   based on a simple intent heuristic (keywords like "overall"/"arc" →
   Tier 1; proper-noun-heavy → Tier 3).
4. **Slice raw text by `scene_index`** in `_fetch_raw_text` instead of
   reading whole `.md` files — scene boundaries (`---`) already exist.
5. **Split citation label vs. citation metadata.** Keep the user-visible
   label clean (arc / episode / part / scene). Carry `file_path`,
   `scene_index`, and canonical story order in metadata for
   verification. Don't spam file paths into every answer.
6. **Cache expensive clients** as module-level singletons:
   `chromadb.PersistentClient`, the collection handle, `GoogleModel`,
   `genai.Client`. Currently rebuilt on every call.
7. **Make model names configurable** via env vars with sensible
   defaults: `LINKURA_CHAT_MODEL`, `LINKURA_EMBEDDING_MODEL`,
   `LINKURA_CHROMA_DB_PATH`. `gemini-3-flash-preview` and
   `gemini-embedding-2` are preview SKUs hardcoded in `database.py`.
8. **Tighten State Ledger injection.** Inject only ledger slices for
   arcs the question is actually about (from query analysis), not for
   every arc that happened to appear in retrieval. Drop `indent=2` on
   the JSON dump to cut tokens.
9. **Soften the system prompt's "strictly raw source text" claim** until
   the evidence gate in the next section is in place.

## High-impact pipeline recommendations

These change the shape of the pipeline. Ordering roughly by dependency.

10. **Index Tier 4 scenes.** Embed every raw scene with
    `summary_level=4` and full metadata (arc, episode, part,
    scene_index, canonical story order, parent part/episode/year IDs,
    file_path, detected speakers, `is_prose`). Either extend
    `story_nodes` or add a `story_scenes` collection. Biggest single
    retrieval fix.
11. **Gate factual answers on raw evidence.** Implement the fallback
    flow from the prompt-mismatch section above. Even before Tier 4
    lands, expand from Tier-3 summary hits to the raw scenes of the
    matched part. This is priority #2 of the whole plan — until the
    system refuses to answer from summaries alone, the "strictly raw
    source text" prompt is a lie.
12. **Raise candidate recall, narrow late.** Current `n_results=3` is
    fragile for multi-hop, character-history, aliases/honorifics,
    "when did X first happen," and any question where the signal is a
    small detail in a long part. **Starting points** (to tune against
    the eval set, not constants):
    - top 20–50 summary records for routing
    - top 30–100 raw scene records for direct evidence
    - final 5–12 chunks after reranking
    
    Final prompt stays small; retrieval stage needs higher recall.
13. **Neighbor-scene expansion.** After a raw-scene hit, pull adjacent
    scenes from the same part. Narrative evidence often spans scene
    boundaries (setup → key dialogue → resolution). **Bound to ±1–2
    scenes** and rerank with the original candidates — unbounded
    expansion blows up the prompt.
14. **Hierarchical routing with RRF.** Run queries per tier (or per
    intent) plus one against raw scenes, fuse with Reciprocal Rank
    Fusion, then fan out from Tier-1/2 hits to children and re-score.
    RRF fits because it combines multiple ranking sources (dense,
    lexical, per-tier) without requiring score calibration.
15. **Hybrid retrieval.** Add a lexical side-index (SQLite FTS,
    `rank_bm25`, tantivy, or Whoosh) over both English summaries and
    raw Japanese scenes. Essential for proper nouns (character / unit /
    place names), exact episode numbers, and quoted dialogue — things
    dense English embeddings silently drop.
16. **Query expansion via glossary.** Before the dense query, expand
    the user question with Japanese/English aliases from
    `glossary.json` (e.g., "Kaho" → "Kaho Hinoshita / 花帆 /
    日野下花帆"). Cheap and high-impact given the bilingual corpus.
17. **Re-ranker.** One Gemini Flash call scoring (question, chunk)
    pairs over top candidates before passing top-k to the answerer.
    Useful reranking signals: exact character/name matches, glossary
    alias matches, episode/arc constraints, dialogue-from-mentioned-
    speakers, proximity to other high-scoring chunks, summary parent
    score.
18. **Query analysis / routing.** Lightweight first pass that extracts
    arc/episode constraints, side-vs-main, character names and
    aliases, temporal phrases ("before episode 12", "as of this
    point"), and the intent bucket (summary vs. exact evidence vs.
    comparison vs. chronology vs. quantitative). Drives metadata
    filters and tier choice.
19. **Canonical story-order field (PLAN).** Stamp every node at ingest
    with a numeric canonical ordering independent of filesystem
    traversal. Must respect the main-vs-side-story interleaving
    `episode_sort_key` already encodes (side stories slot between
    104 Main and 105 Main). **Do not** rely on string comparisons like
    `arc_id <= target` — string compare breaks when arc IDs widen or
    when side stories interleave. Use dedicated numeric fields.
20. **Move story ordering to a manifest/config.** `episode_sort_key`
    hardcodes arc IDs 103/104/105 and where side stories slot in.
    Ordering should come from a manifest or generated canonical-order
    table so adding a new arc does not require editing summarization
    code. Pairs naturally with item 19.
21. **Stale-vector pruning.** Chroma upsert never deletes. If a
    markdown file is renamed, deleted, or re-split, the old vectors
    silently remain and contaminate retrieval. Track IDs per ingest
    run (preferably from the manifest in item 23) and delete any that
    didn't reappear. First-class concern, not a cleanup.

## Tunable knobs

The following should all be driven empirically against the retrieval
eval set (item 25), not guessed:

- candidate counts at each stage (routing, raw evidence, final)
- rerank top-k cutoff
- neighbor-scene expansion width (start at ±1–2)
- hybrid dense/lexical weighting
- RRF constant (`k` in `1/(k+rank)`)
- intent-classifier thresholds

Without the eval harness, each of these is a shot in the dark.

## Data-layer upgrades

22. **Preserve structured dialogue and narrative beats.**
    `parse_script_line` output is currently thrown away. Store
    structured (speaker, text) turns for script files; tag scene
    metadata with detected speakers. For prose files, store quoted
    dialogue separately from narrative beats so both can be queried
    independently. Enables speaker-specific retrieval, "who said X?"
    queries, dialogue-vs-narrative filtering, character co-occurrence,
    cleaner citations, **and code-based counting for quantitative
    questions** (item 28).
23. **Version the index and cache.** Today `summaries_cache.json` keys
    on hierarchy only, so renaming a file, changing a prompt, swapping
    the glossary, or upgrading the model can leave stale generated
    summaries in Chroma while ingestion reports success. Add an
    ingestion manifest: source file hashes, parser version,
    summarization prompt version, glossary hash, chat/embedding model
    names, schema version, timestamp. Drives cache invalidation and
    the stale-vector pruning in item 21.
24. **Give State Ledger facts provenance, extracted from raw scenes.**
    Current `extractor.py` merges from summaries and is lossy — role
    is replaced by the longer string, honorifics overwrite per target,
    `is_active` is OR'd forever, locations and groups are flat sets.
    Replace with source-backed fact records:
    `(subject, predicate, object, arc, episode, part, scene,
    valid_from, valid_to, confidence, extracted_quote)`. **Extract
    from raw scenes, not generated summaries** — extracting from
    summaries compounds summarization loss and destroys provenance.
    Final answers still cite source chunks; the ledger becomes
    routing/consistency context.

## Test coverage

25. **Retrieval eval harness.** ~30 golden-set questions with expected
    arc/episode/scene sources. Track recall@k for expected scenes,
    reranker hit rate, answer faithfulness, citation correctness,
    temporal leakage, glossary consistency. All the tunable knobs
    above depend on this.
26. **Pipeline unit coverage.** Current tests cover package import,
    scene splitting, script detection, and hierarchy extraction only.
    Extend to:
    - ingestion end-to-end (file → scene → summary → Chroma upsert)
    - Chroma metadata shape (required fields, types)
    - citation formatting (round-trip from metadata → label → parse)
    - state extraction (fact extraction deterministic on fixed inputs)
    - answer-grounding smoke tests (retrieved chunks contain claimed
      evidence)
    - cache invalidation behavior (manifest change → rebuild)

## Roadmap — do AFTER the base pipeline is solid

These compound on retrieval quality. Running them over a weak index
just amplifies bad retrieval — land items 10–21 and the eval harness
first.

27. **Agentic query loop.** Move `StoryQueryEngine` to a tool-calling
    agent (`pydantic-ai` is already in deps). Tools:
    - `search(query, tier?, arc_id?)`
    - `get_scene(path, scene_index)`
    - `get_state(arc_id, as_of_episode?)`
    - `lookup_glossary(term)`
    
    Fixes multi-hop questions flat RAG can't answer (e.g., "how did
    X's view of Y change from Year 103 to 105").
28. **Quantitative queries over structured data** (PLAN Phase 6.3).
    Questions like "how many times does Kaho speak in Year 103?"
    should count from the structured dialogue table built in item 22,
    not from LLM inference over retrieved text. Depends on speaker
    metadata landing first.
29. **Audit pass (PLAN Phase 5).** Secondary LLM call comparing the
    draft answer against retrieved sources + State Ledger + Glossary;
    flags retcons, wrong honorifics, hallucinated names. Not a
    substitute for better retrieval — source-grounded citations come
    first.

## Small cleanups / latent bugs

- `_node_id` uses `scene_index` but summary nodes set it to `-1`. Not
  serious today because level + episode + part already disambiguate
  summaries — **but this escalates the moment Tier 4 is indexed**
  (item 10), because raw scenes use `scene_index` as their actual
  disambiguator. Fix as part of that change, not independently.
- `summaries_cache.json` is rewritten in full after every part. Moving
  to SQLite or JSONL is lower priority than **content-hash
  invalidation and stale-vector pruning** (items 21, 23). Format
  migration alone fixes nothing.
- Consider streaming for `chat` — `run_sync` blocks the whole answer.
  Not a retrieval-quality issue; lowest priority.

## Target architecture

```text
Markdown story files
        |
        v
Parser / normalizer (structured dialogue + narrative beats)
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
Query analysis (intent, arc/episode scope, speakers, temporal phrases)
        |
        v
Metadata filters + glossary expansion
        |
        v
Summary routing + raw scene retrieval + hybrid search
        |
        v
RRF + reranking + bounded neighbor expansion
        |
        v
Evidence gate: answer only from raw chunks (or report insufficient)
        |
        v
Answer generation with source citations
        |
        v
Optional audit pass (retcon / glossary / honorific checks)
```

## Priority order

1. Index raw scene chunks and retrieve them for Q&A (item 10).
2. Gate factual answers on raw evidence, or soften the prompt until
   raw evidence is available (item 11, plus quick win 9).
3. Set embedding `task_type` + batch calls (items 1–2).
4. Glossary-expanded hybrid retrieval for proper nouns and bilingual
   queries (items 15–16).
5. Raise candidate recall + bounded neighbor expansion + reranking
   (items 12, 13, 17).
6. Hierarchical routing with RRF and child expansion (item 14).
7. Query analysis + metadata filters + canonical numeric story-order
   (items 18–19).
8. Move story ordering to manifest/config (item 20).
9. Index manifest + cache versioning + stale-vector pruning
   (items 21, 23).
10. State Ledger provenance, extracted from raw scenes (item 24).
11. Structured dialogue / narrative beats (item 22).
12. Retrieval eval harness + pipeline unit tests (items 25–26).
13. Agentic loop, quantitative queries, audit pass (items 27–29). Only
    after everything above is solid.

## Bottom line

The current system is a solid prototype for browsing generated
summaries. It is not yet a dependable source-grounded Q&A RAG.

The single highest-impact architectural change is to make summaries a
routing layer and raw scenes the evidence layer. The highest-impact
quick wins are embedding `task_type`, batched embeddings, glossary
query expansion, prompt/evidence gating, and stale-index hygiene.

Once raw chunks are indexed, reranked, cited, and **gated as required
evidence**, every later improvement (hybrid search, temporal filters,
agentic loop, audit pass) becomes additive rather than a rewrite.
