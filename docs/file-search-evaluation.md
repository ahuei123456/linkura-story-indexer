# Gemini File Search for `linkura-story-indexer` — Evaluation

Discussion notes evaluating whether to adopt Google's Gemini File Search
(<https://ai.google.dev/gemini-api/docs/file-search>) and how it would interact
with the existing pipeline.

## 1. What File Search is

Managed RAG service: upload files → auto chunk + embed with the store's
configured embedding model (for example `models/gemini-embedding-2`) → stored
in a `fileSearchStore` → invoke as a `Tool` on `generate_content`. Returns
answers with `grounding_metadata` citations. Storage free; indexing billed at
embed rates; retrieved tokens billed as context.

**Critical architectural constraint:** no standalone retrieval endpoint.
Confirmed in venv (`file_search_stores.py` only exposes CRUD + upload/import,
no `query`/`search`). The only way to use the vectors is via the `Tool` on a
generation call.

## 2. How it maps to the existing pipeline

| Component | Current | File Search equivalent |
|---|---|---|
| Chunker (`indexer/chunker.py`) | Coalesced scenes, 500/1200/1800 char window, `scene_start/end` metadata | Auto whitespace chunking by default; scene boundaries must be preserved by uploading pre-split chunk files and metadata |
| Embeddings (`database.py`) | `gemini-embedding-2` with task prefixes | Same model, prefixes managed |
| Vector store | Chroma `story_nodes`, rich metadata | Managed; custom metadata as typed key/value |
| Retrieval (`query/engine.py`, `lexical.py`) | Semantic + FTS5 BM25 fused via RRF | Semantic only, no fusion possible |
| Intent routing (`query/analysis.py`) | Heuristic classification + constraint extraction | Not provided |
| Summarizer | 3-tier hierarchical, rolling context, hash cache | Not provided |
| Provider abstraction | Google/OpenAI swappable | Gemini-only |

## 3. Metadata support

- Typed custom metadata: `{"key": ..., "string_value"|"numeric_value": ...}`
- Query-time `metadata_filter` using AIP-160 syntax. Equality and conjunction
  are the safest baseline; richer predicates should be validated empirically
  against the API.
- File Search tool knobs include `top_k`, `metadata_filter`, and optional
  vector distance/similarity thresholds in the SDK. These tune retrieval inside
  generation; they do not provide a standalone candidate API.
- Multiple stores; `file_search_store_names` takes a list
- `grounding_metadata` returns document title + custom metadata

## 4. Chunking config — SDK carries no default

`WhiteSpaceConfig.max_tokens_per_chunk` and `max_overlap_tokens` are both
`Optional[int]` with `default=None`
(`.venv/Lib/site-packages/google/genai/types.py:15468-15498`). SDK omits when
unset; server-side default is undocumented.

**Always set `chunking_config` explicitly in production.** Validate the
accepted ceiling empirically (`max_tokens_per_chunk: 2048` is a reasonable
starting point but untested).

## 5. Summary corpus sizing (`summaries_cache.json`)

449 entries, English (~4 chars/token):

| Level | Count | Max chars | Max tokens (≈) |
|---|---|---|---|
| Part | 400 | 5,427 | ~1,360 |
| Episode | 45 | 5,950 | ~1,490 |
| Year | 4 | 7,297 | ~1,825 |

All comfortably one-chunk-able if `max_tokens_per_chunk: 2048` is accepted.

## 6. The "hybrid" trap (File Search as embedding-only replacement)

**Doesn't work.** No retrieval endpoint = can't fuse File Search KNN with
FTS5. Workaround (stub-prompt `generate_content`, harvest
`grounding_metadata`) is wasteful and noisy.

Only clean hybrid: **partition by corpus** — File Search owns one corpus
end-to-end (e.g., summaries for public reader), Chroma keeps the other (raw
scenes with hybrid retrieval). No fusion across them.

## 7. Agentic search interaction

Spectrum:

1. Static heuristics (today: `analysis.py`)
2. **LLM router** — single classification call, dispatch one tool (NOT yet
   agentic)
3. Tool-calling model — model picks tool(s) per turn
4. Agentic loop — model sees results, re-calls, refines

File Search is useful at rung 2, especially as a one-shot summary/raw-corpus
answer tool, but becomes less useful as the system moves toward rung 3/4. It
has tuning knobs (`top_k`, `metadata_filter`, thresholds), but retrieval remains
trapped inside `generate_content` instead of exposing inspectable candidates.
Real agentic value comes from rich, narrowly-typed tools (`search_summaries`,
`search_raw`, `get_scene`, `get_character_state`) — which File Search can't
provide by itself.

**Provider lock-in:** File Search is a Gemini-only `Tool`. OpenAI agent path
can't use it; the custom tool surface gets built regardless.

## 8. Part summaries vs raw Japanese for one-shot RAG

There is no universal winner. Pick the corpus based on the one-shot use case.

For a public-reader conversational backend, **Part summaries** are the cleaner
first experiment because they are dense, English, and already normalized for
broad questions. For source-grounded factual QA, **raw chunks** preserve the
truth better because summaries omit exact evidence.

| Question type | Summaries | Raw JP |
|---|---|---|
| Plot, character, theme, recap | Excellent | Poor |
| Exact dialogue | Useless | Decent |
| Keyword / quote search | Bad | Decent (but FTS5 wins) |
| Linguistic / translation | Useless | Best option |
| Aggregate / count | Mediocre | Bad |

Raw JP is a poor shape for broad English one-shot synthesis: cross-lingual,
lower per-chunk density, and semantic retrieval is not lexical lookup. It is
still the right shape for exact dialogue, source verification, linguistic /
translation questions, and citation-grounded factual answers. To keep it usable,
upload pre-split retrieval chunks with metadata rather than whole parts and
blind server-side chunking.

## 9. Multi-level summaries — redundancy concern

Year ⊃ Episode ⊃ Part content-wise → top-K pollution when surfaced for the
same question. But levels have **different shapes**, not just sizes
(Year = thematic, Episode = arc cause/effect, Part = plot beats), so they
earn their keep for *different* question scopes.

- **No router** → upload Parts only. Avoid redundancy tax.
- **With router** → multi-level via `summary_level` filter OR separate stores.
  Each query hits exactly one tier.
- **Year summaries (only 4)** → too few for retrieval; prepend directly to
  prompt when router flags a thematic query.

## 10. Recommended path

1. **Don't replace Chroma.** It's load-bearing for hybrid retrieval,
   scene-precise citations, OpenAI portability, and the agentic foundation.
2. **Wrap `query/engine.py` modes as typed Pydantic-AI tools.** Foundation for
   routing AND any later agentic loop.
3. **Add File Search as ONE additional tool** over a *separate* corpus — most
   likely Part summaries (and optionally Episode summaries) for the
   "public reader" / broad one-shot conversational use case. Consider a
   separate raw-chunk File Search store only for direct comparison against the
   local raw retrieval path.
4. **Build an LLM router (rung 2)** that picks among the typed tools.
   Replaces heuristics in `analysis.py` with model judgment, returns
   `{tool_name, args}` via structured output.
5. **Year summaries: skip File Search.** Inject directly into prompts when
   scope warrants.
6. **Only climb to rung 3/4 (true agentic)** once router-mode is in production
   and there's data on its misroutes.
