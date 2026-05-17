#!/usr/bin/env bash
set -euo pipefail

cache_file="summaries_cache.json"
output_dir="site/summary-reader"
story_order_file="story_order.yaml"
port="8000"
serve=0

usage() {
  cat <<'EOF'
Usage: scripts/export-summary-reader.sh [options]

Options:
  --cache-file PATH        Path to summaries_cache.json
  --output-dir PATH        Directory to write the static reader site
  --story-order-file PATH  Path to story_order.yaml
  --port PORT              Port for --serve preview
  --serve                  Serve the exported site after building
  -h, --help               Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cache-file)
      cache_file="${2:?--cache-file requires a path}"
      shift 2
      ;;
    --output-dir)
      output_dir="${2:?--output-dir requires a path}"
      shift 2
      ;;
    --story-order-file)
      story_order_file="${2:?--story-order-file requires a path}"
      shift 2
      ;;
    --port)
      port="${2:?--port requires a value}"
      shift 2
      ;;
    --serve)
      serve=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

uv run indexer export-summary-reader \
  --cache-file "$cache_file" \
  --output-dir "$output_dir" \
  --story-order-file "$story_order_file"

echo "Summary reader exported to $output_dir"

if [[ "$serve" -eq 1 ]]; then
  echo "Serving http://localhost:$port"
  uv run python -m http.server "$port" --directory "$output_dir"
fi
