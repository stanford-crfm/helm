#!/bin/bash

# This script fails when any of its commands fail.
pip install -e .
pip check

# Python style checks and linting
black --check --diff src scripts || (
  echo ""
  echo "The code formatting check failed. To fix the formatting, run:"
  echo ""
  echo ""
  echo -e "\tvenv/bin/black src"
  echo ""
  echo ""
  exit 1
)

mypy src scripts
flake8 src scripts

echo "Done."
