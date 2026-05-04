param(
    [string]$CacheFile = "summaries_cache.json",
    [string]$OutputDir = "site/summary-reader-production",
    [string]$StoryOrderFile = "story_order.yaml",
    [string]$SourceDir = "web/summary-reader-production",
    [int]$Port = 8011,
    [switch]$Serve
)

$ErrorActionPreference = "Stop"

uv run indexer export-production-summary-reader `
    --cache-file $CacheFile `
    --output-dir $OutputDir `
    --story-order-file $StoryOrderFile `
    --source-dir $SourceDir

Write-Host "Production summary reader exported to $OutputDir"

if ($Serve) {
    Write-Host "Serving http://localhost:$Port"
    uv run python -m http.server $Port --directory $OutputDir
}
