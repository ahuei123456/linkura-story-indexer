# Implementation Plan: Tiered Story RAG

## 1. Discussion Summary
The goal is to implement a **Hierarchical RAG (Retrieval-Augmented Generation)** system for a story repository (mobile game scripts and prose side stories). This system ensures high accuracy, proper sourcing, and plot consistency by navigating through multiple levels of granularity.

### Core Architecture
* **Tier 1 (Arc):** Global summaries for broad thematic queries.
* **Tier 2 (Volume):** Summaries for mid-level context.
* **Tier 3 (Chapter/File):** Precise summaries of individual story beats.
* **Tier 4 (Scene/Raw Chunks):** Structured dialogue/prose segments (the "Source of Truth").

### Data Handling
* **Script Format:** Stored as structured lists: `(speaker, text)`.
* **Prose Format:** Stored in a hybrid format (Narrative + Dialogue) to preserve internal thoughts and environmental context.
* **State Ledger:** A separate JSON-based "Fact Table" tracking character statuses, locations, and relationships to prevent hallucinations.

---

## 2. Plan of Attack (Repo LLM)

### Phase 1: Ingestion & Parsing
1.  **Scene Splitting:** Parse `.md` files using `---` delimiters as scene boundaries.
2.  **Hybrid Parsing:** * For scripts: Extract `speaker` and `text`.
    * For prose: Use an LLM pass to convert narrative into structured "beats" while identifying implicit speakers.
3.  **Metadata Tagging:** Every chunk must inherit `arc_id`, `vol_id`, `chap_id`, and `scene_index`.

### Phase 2: Bottom-Up Indexing
1.  **Chapter Summarization:** Use the **Refine** method (Summary + New Chunk) to generate Level 3 summaries for all files.
2.  **Volume/Arc Synthesis:** Recursively summarize Level 3 -> Level 2 -> Level 1.
3.  **Vector Upsert:** Store embeddings in a Vector DB (e.g., Pinecone) with hard relational links between tiers.

### Phase 3: The State Ledger
1.  **Extraction:** Run a final pass over Arc summaries to generate a `world_state.json`.
2.  **Contents:** Track alive/dead status, current locations, and character honorifics (e.g., Kaho uses "-chan").

### Phase 4: Retrieval Logic (Orchestration)
1.  **Intent Detection:** Determine if the query is Global (Arc) or Specific (Scene).
2.  **Temporal Filtering:** Apply metadata filters to prevent the LLM from "knowing the future" during translation or specific QA.
3.  **Tool Calling:** * `search_summaries(query)`: Finds the relevant Chapter/Volume.
    * `get_detailed_context(chapter_id)`: Pulls raw structured scenes.
    * `get_state_ledger()`: Injects the "Current Truth" into the system prompt.

### Phase 5: Verification & Translation
1.  **Constraint Injection:** Force the LLM to use the "Mandatory Glossary" and "State Ledger" in its system prompt.
2.  **Audit Loop:** A secondary LLM pass compares generated output against the Index to flag "Retcon" errors or character name inconsistencies.