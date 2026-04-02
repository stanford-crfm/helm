#!/usr/bin/env bash
# Full local Jekyll preview (matches CI): expand MkDocs-sourced pages, then serve.
# Restores docs/*.md sources from git on exit so you do not commit expanded HTML dumps.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DOCS_PAGES=(
  docs/models.md
  docs/metrics.md
  docs/scenarios.md
  docs/perturbations.md
  docs/schemas.md
)

restore_docs_sources() {
  git checkout -- "${DOCS_PAGES[@]}" 2>/dev/null || true
}

trap restore_docs_sources EXIT INT TERM HUP

python3 docs/scripts/jekyll_prepare_mkdocstring_pages.py

cd docs
bundle exec jekyll serve --livereload "$@"
