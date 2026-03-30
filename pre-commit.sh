#!/bin/bash

# This script fails when any of its commands fail.
set -e

if command -v uv >/dev/null 2>&1 && [[ -f .venv/bin/activate ]]; then
  echo 'Running pre-commit with uv...'
  source ".venv/bin/activate"
else
  echo 'Running pre-commit without uv...'
  pip check
fi

# Python style checks and linting
black --check --diff src scripts || (
  echo ""
  echo ""
  echo "The code formatting check failed. To fix the formatting, run:"
  echo ""
  echo -e "\tblack src scripts"
  echo -e "\tgit commit -a"
  echo ""
  exit 1
)

mypy src scripts

flake8 src scripts

echo "Done."
