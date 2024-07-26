#!/bin/bash

set -e

yarn --version || (
  echo ""
  echo "Failed: yarn not found"
  echo "Please install yarn (see: https://classic.yarnpkg.com/lang/en/docs/install/) "
  echo "and then run 'yarn install' from helm-frontend/ to fix this."
  exit 1
)

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
