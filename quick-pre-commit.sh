#!/bin/bash

# Quick version of pre-commit.sh to run during development.
# This script fails when any of its commands fail.
set -e

# Python style checks and linting
black src scripts
mypy src scripts
flake8 src scripts

echo "Done."
