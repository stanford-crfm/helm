#!/bin/bash

set -e

FRONTEND_DIR="$(dirname "$0")/helm-frontend"
cd "$FRONTEND_DIR"

if ! yarn --version &> /dev/null; then
  echo ""
  echo ""
  echo "FAILED: Yarn is not installed"
  echo "Follow the instructions at https://yarnpkg.com/getting-started/install to install Yarn."
  echo ""
  exit 1
fi

if [ ! -f "yarn.lock" ]; then
  echo ""
  echo ""
  echo "FAILED: Frontend dependencies are not installed. To install frontend dependencies, run:"
  echo ""
  echo -e "\tcd $FRONTEND_DIR"
  echo -e "\tyarn install"
  echo ""
  exit 1
fi

yarn format:check || (
  echo ""
  echo ""
  echo "FAILED: The frontend code formatting check failed. To fix the formatting, run:"
  echo ""
  echo -e "\tcd $FRONTEND_DIR"
  echo -e "\tyarn format"
  echo -e "\tgit commit -a"
  echo ""
  exit 1
)

yarn lint:check || (
  echo ""
  echo ""
  echo "FAILED: The frontend code linter check failed. To lint the frontend code, run:"
  echo ""
  echo -e "\tcd $FRONTEND_DIR"
  echo -e "\tyarn lint"
  echo -e "\tgit commit -a"
  echo ""
  echo "You may have to manually fix some errors if the automatic linter fails."
  echo ""
  exit 1
)
