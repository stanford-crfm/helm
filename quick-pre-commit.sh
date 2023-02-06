#!/bin/bash

# Quick version of pre-commit.sh to run during development.
# This script fails when any of its commands fail.
set -e

# Python style checks and linting
black src scripts
flake8 src scripts
mypy --install-types --non-interactive src scripts

echo "Done."
