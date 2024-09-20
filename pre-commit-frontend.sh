#!/bin/bash

set -e

cd "$(dirname "$0")/helm-frontend"

if ! yarn --version &> /dev/null; then
  echo ""
  echo "Failed: Yarn is not installed"
  echo "Follow the instructions at https://yarnpkg.com/getting-started/install to install Yarn."
  echo ""
  exit 1
fi

if [ ! -f "yarn.lock" ]; then
  echo ""
  echo "Failed: Frontend is not installed."
  echo "Run \`yarn install\` to install the frontend."
  echo ""
  exit 1
fi

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
