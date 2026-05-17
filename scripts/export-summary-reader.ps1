param(
    [string]$CacheFile = "summaries_cache.json",
    [string]$OutputDir = "site/summary-reader",
    [string]$StoryOrderFile = "story_order.yaml",
    [int]$Port = 8000,
    [switch]$Serve
)

$ErrorActionPreference = "Stop"

uv run indexer export-summary-reader `
    --cache-file $CacheFile `
    --output-dir $OutputDir `
    --story-order-file $StoryOrderFile

Write-Host "Summary reader exported to $OutputDir"

if ($Serve) {
    Write-Host "Serving http://localhost:$Port"
    uv run python -m http.server $Port --directory $OutputDir
}
