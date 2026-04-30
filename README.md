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
`LINKURA_CHAT_MODEL`, then the repo default Gemini chat model. When
`LINKURA_INGEST_PROVIDER=openai`, set `LINKURA_INGEST_MODEL` or set
`LINKURA_CHAT_MODEL` to an OpenAI model name so the default Gemini model is not
sent to OpenAI. Google generation requires `GOOGLE_API_KEY`. OpenAI generation
requires `OPENAI_API_KEY`.

Embeddings are separate and still use the Google GenAI embedding path controlled
by `LINKURA_EMBEDDING_MODEL` (default: `gemini-embedding-2`). Running `ingest`
therefore always requires `GOOGLE_API_KEY`, even when
`LINKURA_INGEST_PROVIDER=openai`.

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
