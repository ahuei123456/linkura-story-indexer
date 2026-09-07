# linkura-story-indexer
Scripts to index the LLLL story

## Development direction

The active roadmap is now a source-browsing agent harness: canonical browsing,
literal/regex text search, deliberate source reading, and source-backed answers,
with no embedding or Chroma dependency in the new path. See [PLAN.md](PLAN.md)
and [TASKS.md](TASKS.md). Further vector tuning, reranking, and managed File
Search integration are cancelled.

Local corpus preparation and `StoryCorpus` reads work without embeddings or
Chroma. The investigative harness is still planned. The existing vector runtime
is retained for comparison through explicit `--legacy` selection on `ingest`,
`query`, `chat`, and `eval run`; those commands retain their embedding requirements.

## Generation Models

Summary generation and `extract-state` world-state generation use the
configured generation provider:

- `LINKURA_INGEST_PROVIDER=google|openai` (default: `google`)
- `LINKURA_INGEST_MODEL=<model name>`
- `OPENAI_BASE_URL=<compatible endpoint>` (optional, for OpenAI-compatible Chat
  Completions endpoints)

If `LINKURA_INGEST_MODEL` is unset, generation falls back to
`LINKURA_CHAT_MODEL`, then the repo default Gemini chat model. Query answer
generation uses the same configured provider and model, and LLM routing uses
the same provider. When
`LINKURA_INGEST_PROVIDER=openai`, set `LINKURA_INGEST_MODEL` or set
`LINKURA_CHAT_MODEL` to an OpenAI model name so the default Gemini model is not
sent to OpenAI. Google generation requires `GOOGLE_API_KEY`. OpenAI generation
requires `OPENAI_API_KEY`.

Embeddings are separate and still use the Google GenAI embedding path controlled
by `LINKURA_EMBEDDING_MODEL` (default: `gemini-embedding-2`). Running `ingest`
and retrieval through `query` or `chat` therefore always requires
`GOOGLE_API_KEY`, even when `LINKURA_INGEST_PROVIDER=openai`.

## Querying

With `--legacy`, `query` and `chat` use raw hybrid retrieval by default. The query
path retrieves raw source evidence with `summary_level: 4`, does not add
analyzer-derived metadata filters, and does not perform summary-tier retrieval
or summary fanout.

Use `--routing-mode heuristic` when you want targeted scoped workflows that
apply analyzer-derived filters or structured helpers. Use
`--routing-mode llm_router` to let the configured router model select one
typed query tool before retrieval. Use `--routing-mode agentic` for a capped,
multi-step tool-calling loop that can combine glossary lookup, retrieval,
point-in-time State Ledger checks, and exact SQL dialogue counts. Counting
questions in agentic mode must use `count_dialogue`; the answer is never an
LLM estimate. Agentic mode uses at most eight model requests by default; set
`LINKURA_AGENT_REQUEST_LIMIT` to change the cap. The router model is controlled
by `LINKURA_ROUTER_MODEL`. With the Google provider it defaults to
`gemini-3.1-flash-lite-preview`; with the OpenAI provider it defaults to the
configured generation model.

```powershell
indexer query --legacy "What happened in the 105th term?" --routing-mode heuristic
indexer chat --legacy --routing-mode llm_router
indexer query --legacy "How many dialogue turns does 花帆 have in 103?" --routing-mode agentic
```

Add `--audit` to `query` or `chat` for a secondary answer check against the
retrieved evidence, State Ledger, and official Glossary. Audit is opt-in for
interactive use and reports possible retcons, wrong honorifics, or
hallucinated names without changing the draft answer.

```powershell
indexer query --legacy "What does 花帆 call Sayaka in episode 1?" --routing-mode agentic --audit
indexer chat --legacy --routing-mode agentic --audit
```

Use `--routing-mode off` to make the default explicit:

```powershell
indexer query --legacy "What happened in the 105th term?" --routing-mode off
indexer chat --legacy --routing-mode off
```

### Answer prompts

Static answer-generation instructions are Markdown resources in
`src/linkura_story_indexer/prompts`. Edit those files to revise answer policy;
keep questions, retrieved context, glossary entries, and State Ledger facts as
dynamic renderer inputs. Increment `PROMPT_VERSION` in the prompt module for a
behavioral change so evaluations and future traces can identify the revision.
When `summaries_cache.json` is present, every valid Year-level summary is also
loaded into the system prompt as a generated Story Overview. These summaries
can ground broad Year/Arc synthesis; narrower claims should continue drilling
down to Episode, Part, or raw-scene evidence.

Run the prompt and query-engine tests after a prompt change:

```powershell
uv run pytest tests/test_prompts.py tests/test_query_engine.py
```

## Retrieval Evaluation

Run the checked-in retrieval golden set without answer generation:

```powershell
uv run indexer eval run --legacy --golden-set eval/golden_questions.json --routing-mode off --output runs/baseline.json
```

Use `--routing-mode heuristic` to compare analyzer-derived filters,
`--routing-mode llm_router` to evaluate typed router dispatch, or
`--routing-mode agentic --answer-mode` to evaluate the multi-step agent. Add
`--audit` to the answer-mode run to aggregate audit cleanliness and flag counts.
The audit flag requires answer mode. Compare router and agentic runs with the
same golden set:

```powershell
indexer eval run --legacy --routing-mode llm_router --answer-mode --output runs/router.json
indexer eval run --legacy --routing-mode agentic --answer-mode --output runs/agentic.json
indexer eval diff runs/router.json runs/agentic.json
```

Reranker metrics in legacy runs remain unavailable; the reranker is cancelled.
The current golden set has 30 single-source questions. The replacement harness
benchmark in TASKS.md measures answer correctness, completeness, source support,
scope compliance, and cost/time across multi-source narrative questions.

## Index Rebuilds

Build a local source catalog and lexical index without generation or embeddings:

```powershell
uv run indexer prepare-corpus --story-dir story --output-dir prepared_corpus
```

To generate optional summaries, configure the generation provider above and run:

```powershell
uv run indexer summarize-corpus --story-dir story --cache-file summaries_cache.json
uv run indexer prepare-corpus --story-dir story --output-dir prepared_with_summaries --summaries summaries_cache.json
```

`summarize-corpus` uses the existing hierarchical summarizer and its cache
fingerprints, without embedding or Chroma access. It generates missing or
changed summaries, retains generation context, and removes cache entries that
are no longer emitted by the current corpus. The cache is published only after
successful generation. Generation is an explicit API operation; preparation and
reading remain local. Both commands accept `--story-order` and `--glossary`.

To rebuild an existing prepared directory safely, opt into immutable replacement:

```powershell
uv run indexer prepare-corpus --story-dir story --output-dir prepared_corpus --replace
```

Replacement retains the legacy files and publishes each new generation under
`.revisions`, then atomically updates `current.json`. Pointer-aware readers use
`StoryCorpus.from_publication(...)`; direct paths continue to read the legacy
files for compatibility.

Read a first preparation with `StoryCorpus.from_prepared_dir`; after publishing
with `--replace`, use `from_publication` to select the current generation:

```python
from linkura_story_indexer.corpus import StoryCorpus

corpus = StoryCorpus.from_prepared_dir("prepared_corpus")
scenes = corpus.iter_scenes()
summary = corpus.get_summary("YEAR|103")  # None when no summary was imported
glossary_entry = corpus.lookup_glossary("Kaho")
order = corpus.story_order
```

These reads use the prepared source records and metadata snapshots. They require
no API keys and do not start a query engine, generation model, or embedding client.

For references that will be saved or passed between readers, use
`StoryCorpus.scene_reference(file_path, scene_index)` and
`get_scene_by_reference(reference)`. A reference includes its corpus revision;
a reader from a different revision rejects it, including after a rename or scene
re-splitting. An already-open reader can still resolve its original reference.
To refresh a reference, look up the intended scene in the new reader and create
a new reference explicitly. Bare `get_scene` calls are lookups within that reader,
not persistent references. Pagination belongs to the later browsing-tool work.

Imported summaries report overall `freshness: stale` when a checked source,
rolling predecessor, or direct aggregate-child dependency has changed. Otherwise
overall freshness remains `unverified`. Separate `source_freshness`,
`rolling_freshness`, and `aggregate_freshness` fields describe those scoped checks;
they do not certify complete or transitive dependency freshness. Cache inputs and
fingerprints retain generation provenance. Raw scenes remain available when
summaries are stale or absent.

Without `--replace`, the output directory must be new. Existing source and lexical databases are never replaced in place; `--replace` publishes a retained immutable generation and updates the pointer.

Raw evidence is embedded as coalesced retrieval chunks over adjacent source
scenes. The original parsed scene indexes remain in metadata as
`scene_start`, `scene_end`, and `source_scene_count` for citations.

For the legacy retrieval path, changing retrieval chunk thresholds or raw
metadata schema requires a rebuild or a complete ingest with pruning. Ingest
already prunes stale vector and lexical records by default; --no-prune disables
that cleanup. Older records can otherwise remain active beside new chunk IDs.

Invoke the retained vector ingestion explicitly with `uv run indexer ingest
--legacy`. `query`, `chat`, and `eval run` likewise require `--legacy` before any API or
database initialization. The new investigative query/chat harness is not yet
implemented; omitting the flag does not fall back to vectors.

Changing the embedding model or embedding input format also requires rebuilding
the Chroma index. The default `gemini-embedding-2` path uses inline retrieval
instructions (`title: ... | text: ...` for documents and
`task: search result | query: ...` for queries), which should not be mixed with
older vectors created from raw text or from a different embedding model.

## Summary Reader

Export the cached Year, Episode, and Part summaries to a static reader:

```powershell
.\scripts\export-summary-reader.ps1
```

```bash
./scripts/export-summary-reader.sh
```

Preview locally with:

```powershell
.\scripts\export-summary-reader.ps1 -Serve -Port 8000
```

```bash
./scripts/export-summary-reader.sh --serve --port 8000
```

Deployment notes for self-hosting and GitHub Pages are in
`docs\summary-reader-deploy.md`.

The original production page from `Linkura Summaries.zip` is also preserved
under `web\summary-reader-production`. Export that variant with:

```powershell
.\scripts\export-production-summary-reader.ps1
```

```bash
./scripts/export-production-summary-reader.sh
```
