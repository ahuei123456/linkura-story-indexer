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
typed query tool before retrieval. Use `--routing-mode agentic` for a capped,
multi-step tool-calling loop that can combine glossary lookup, retrieval,
point-in-time State Ledger checks, and exact SQL dialogue counts. Counting
questions in agentic mode must use `count_dialogue`; the answer is never an
LLM estimate. Agentic mode uses at most eight model requests by default; set
`LINKURA_AGENT_REQUEST_LIMIT` to change the cap. If the agent reaches the cap (or
the model errors) before writing an answer, the engine answers from the evidence
it gathered with one normal answer call, preferring scenes the agent opened
directly, then search hits, then summaries. Only when it gathered nothing usable
does it return the raw tool output. The trace records which path answered in
the agent stage's `answer_source` (`agent`, `evidence_synthesis` or
`tool_fallback`). The router model is controlled
by `LINKURA_ROUTER_MODEL`. With the Google provider it defaults to
`gemini-3.1-flash-lite-preview`; with the OpenAI provider it defaults to the
configured generation model.

```powershell
indexer query "What happened in the 105th term?" --routing-mode heuristic
indexer chat --routing-mode llm_router
indexer query "How many dialogue turns does 花帆 have in 103?" --routing-mode agentic
```

Add `--audit` to `query` or `chat` for a secondary answer check against the
retrieved evidence, State Ledger, and official Glossary. Audit is opt-in for
interactive use and reports possible retcons, wrong honorifics, or
hallucinated names without changing the draft answer.

```powershell
indexer query "What does 花帆 call Sayaka in episode 1?" --routing-mode agentic --audit
indexer chat --routing-mode agentic --audit
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

Use `--routing-mode heuristic` to compare analyzer-derived filters,
`--routing-mode llm_router` to evaluate typed router dispatch, or
`--routing-mode agentic --answer-mode` to evaluate the multi-step agent. Add
`--audit` to the answer-mode run to aggregate audit cleanliness and flag counts.
The audit flag requires answer mode. Compare router and agentic runs with the
same golden set:

```powershell
indexer eval run --routing-mode llm_router --answer-mode --output runs/router.json
indexer eval run --routing-mode agentic --answer-mode --output runs/agentic.json
indexer eval diff runs/router.json runs/agentic.json
```

Until the reranker lands, reranker metrics are emitted as unavailable. The
current golden set remains single-scene; multi-hop and expanded quantitative
coverage should be added in a follow-up evaluation issue.

## Answer Benchmark

The retrieval golden set checks whether one gold scene is retrieved. The answer
benchmark in `eval/answer_cases.json` checks whether the final answer is right.
Each case lists the claims a correct answer must make (each with one or more
acceptable source passages), plausible wrong claims that fail the case, an
optional spoiler boundary, and a grading rubric. Cases record whether they came
from an observed failure or were curated, and stay `draft` until a person has
reviewed them against the source.

Two scorers run on every case:

- **Evidence (deterministic):** whether each required passage was seen by the
  run and reached the answer step, whether cited labels exist in the supplied
  evidence, and whether evidence crossed the spoiler boundary.
- **Answer (model-judged):** a judge from a different provider than the system
  under test (default `openai:gpt-6.1-sol`) marks each claim present, partial,
  missing or contradicted, flags forbidden and unsupported statements, and
  checks abstention. A case passes when every claim is present, nothing is
  contradicted or forbidden, and abstention matches the expectation. An
  agentic run that returned raw tool output instead of an answer fails without
  being judged.

Runs record the models, prompt versions, corpus and code revisions, latency,
token usage and an estimated cost from the list prices in
`src/linkura_story_indexer/eval/usage.py`. Models without a configured price
report cost as unavailable. Embedding calls are not metered.

```powershell
uv run indexer eval answers validate
uv run indexer eval answers run --mode all --output-dir runs/answers/2026-11-07-baseline
uv run indexer eval answers run --mode agentic --case a001_kaho_sayaka_first_meeting --no-judge
uv run indexer eval answers rescore runs/answers/2026-11-07-baseline/*.json
uv run indexer eval answers report runs/answers/2026-11-07-baseline/off.json runs/answers/2026-11-07-baseline/agentic.json --output report.md
```

### Reviewing cases

`indexer eval answers review` serves a local page (127.0.0.1 only) for working
through the suite:

```powershell
uv run indexer eval answers review
```

- Edit any field, add or remove claims, sources and forbidden claims, and
  create new cases. Saves are validated against the schema and the corpus
  before `eval/answer_cases.json` is rewritten.
- **Approve** marks a case `reviewed`. Changing a reviewed case's content puts
  it back to `draft`, so approvals always refer to what is in the file.
- The **Source** tab shows each cited passage with the quote highlighted, with
  buttons to widen the context. **Search story** runs a regex over the corpus
  and can add a hit to a claim as a new source.
- **Run results** shows how every saved run answered the case, with the judge's
  per-claim verdicts. The **Runs** view compares all runs found under
  `runs/answers` (change with `--runs`), so new benchmark runs appear there
  after **Reload runs**.

Live runs need `GOOGLE_API_KEY` (query embeddings and Gemini generation) and
`OPENAI_API_KEY` for the default judge. Use `--reviewed-only` to score only
reviewed cases. `rescore` reapplies the deterministic scoring rules to saved
runs after a rule change, without new model calls.

## Index Rebuilds

Raw evidence is embedded as coalesced retrieval chunks over adjacent source
scenes. The original parsed scene indexes remain in metadata as
`scene_start`, `scene_end`, and `source_scene_count` for citations.

Changing retrieval chunk thresholds or raw metadata schema changes the emitted
record IDs. `ingest` prunes Chroma and lexical-index records that the current
run did not emit (on by default; disable with `--no-prune`), so older one-scene
records do not remain active beside the new chunk IDs.

Changing the embedding model or embedding input format also requires rebuilding
the Chroma index. The default `gemini-embedding-2` path uses inline retrieval
instructions (`title: ... | text: ...` for documents and
`task: search result | query: ...` for queries), which should not be mixed with
older vectors created from raw text or from a different embedding model.

## Summary Quality

`ingest` fact-checks each generated Part summary with a second call to the same
model, which compares the draft against the Japanese source and corrects claims
the source does not support (merged statements, details filled in from general
knowledge, misidentified characters). Disable it with `--no-verify-summaries`;
changing the setting invalidates cached Part summaries. Transient provider
errors (rate limits, overloads, timeouts) are retried with backoff; exhausted
credits (HTTP 402) stop the run immediately.

To compare models or prompt changes on a sample of Parts without touching the
cache, use `scripts/compare_summary_models.py`. It writes a side-by-side report
to `.tmp/summary-comparison/`:

```powershell
uv run python scripts/compare_summary_models.py --model google:gemini-3.8-flash --verify --part "104|Main|第8話『Not a marionette』|2" --random 6 --seed 7 --repeat 3
```

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

## Roadmap

Planned work is tracked in GitHub issues; start with the
[agent harness roadmap (#80)](https://github.com/ahuei123456/linkura-story-indexer/issues/80).
`PLAN.md` is the original design document and is kept for background.
