#!/bin/bash

# Run this script before you commit: ./pre-commit.sh

# This script fails immediately when any of its commands fail.
set -e

if ! [ -e venv ]; then
  python3 -m pip install virtualenv
  python3 -m virtualenv -p python3 venv
fi

venv/bin/pip install -r requirements.txt
venv/bin/pip check

# Python style checks and linting
venv/bin/black src  # Mutates code!
venv/bin/mypy src
venv/bin/flake8 src

echo "Done."