# linkura-story-indexer
Scripts to index the LLLL story

## Generation Models

Ingest summarization and `extract-state` world-state generation use the
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

By default, `query` and `chat` use raw hybrid retrieval only. The default query
path retrieves raw source evidence with `summary_level: 4`, does not add
analyzer-derived metadata filters, and does not perform summary-tier retrieval
or summary fanout.

Use `--routing-mode heuristic` when you want targeted scoped workflows that
apply analyzer-derived filters or structured helpers. Use
`--routing-mode llm_router` to let the configured router model select one
typed query tool before retrieval. The router model is controlled by
`LINKURA_ROUTER_MODEL`. With the Google provider it defaults to
`gemini-3.1-flash-lite-preview`; with the OpenAI provider it defaults to the
configured generation model.

```powershell
indexer query "What happened in the 105th term?" --routing-mode heuristic
indexer chat --routing-mode llm_router
```

Use `--routing-mode off` to make the default explicit:

```powershell
indexer query "What happened in the 105th term?" --routing-mode off
indexer chat --routing-mode off
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
uv run indexer eval run --golden-set eval/golden_questions.json --routing-mode off --output runs/baseline.json
```

Use `--routing-mode heuristic` to compare analyzer-derived filters, or
`--routing-mode llm_router` to evaluate typed router dispatch. Until the
reranker lands, reranker metrics are emitted as unavailable.

## Index Rebuilds

Raw evidence is embedded as coalesced retrieval chunks over adjacent source
scenes. The original parsed scene indexes remain in metadata as
`scene_start`, `scene_end`, and `source_scene_count` for citations.

Changing retrieval chunk thresholds or raw metadata schema requires rebuilding
the Chroma index, or pruning stale vectors once Task 9 stale-vector pruning
exists. Otherwise older one-scene records can remain active beside the new
chunk IDs.

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
