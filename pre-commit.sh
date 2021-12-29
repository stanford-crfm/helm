#!/bin/bash

# This script fails when any of its commands fail.
set -e

if ! [ -e venv ]; then
  python3 -m pip install virtualenv
  python3 -m virtualenv -p python3 venv
fi

venv/bin/pip install -r requirements.txt
venv/bin/pip check

# Python style checks and linting
set +e
venv/bin/black --check --diff src
exit_code=$?
if [ $exit_code -ne 0 ]; then
  echo ""
  echo "The code formatting check failed. To fix the formatting, run:"
  echo ""
  echo ""
  echo "    venv/bin/black src"
  echo ""
  echo ""
  exit $exit_code
fi
set -e

venv/bin/mypy src
venv/bin/flake8 src

echo "Done."