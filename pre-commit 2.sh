#!/bin/bash

# This script fails when any of its commands fail.
set -e

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

mypy --install-types --non-interactive src scripts

PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
if [[ $PYTHON_VERSION == 3.12* ]]; then
  echo "Skipping flake8 because Python version is 3.12."
  echo "See https://github.com/stanford-crfm/helm/issues/3072 for more information."
else
  flake8 src scripts
fi

echo "Done."
