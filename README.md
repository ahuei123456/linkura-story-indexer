# linkura-story-indexer
Scripts to index the LLLL story

## Index Rebuilds

Raw evidence is embedded as coalesced retrieval chunks over adjacent source
scenes. The original parsed scene indexes remain in metadata as
`scene_start`, `scene_end`, and `source_scene_count` for citations.

Changing retrieval chunk thresholds or raw metadata schema requires rebuilding
the Chroma index, or pruning stale vectors once Task 9 stale-vector pruning
exists. Otherwise older one-scene records can remain active beside the new
chunk IDs.
