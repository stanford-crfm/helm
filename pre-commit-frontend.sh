#!/bin/bash

set -e

cd "$(dirname "$0")/helm-frontend"
yarn format:check || (
  echo ""
  echo "Failed: 'yarn format:check'"
  echo "Run 'yarn format' from helm-frontend/ to fix these errors."
  echo ""
  exit 1
)

yarn lint:check || (
  echo ""
  echo "Failed: 'yarn lint:check'"
  echo "Run 'yarn lint' from helm-frontend/ to attempt to automatically fix these errors."
  echo "You may have to manually fix some errors."
  echo ""
  exit 1
)
