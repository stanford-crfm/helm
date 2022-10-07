#!/bin/bash

# This script fails when any of its commands fail.
set -e

# Check that python version is at least 3.8.
valid_version=$(python3 -c 'import sys; print(sys.version_info[:2] >= (3, 8))')
if [ "$valid_version" == "False" ]; then
  echo "Python 3 version (python3 --version) must be at least 3.8, but was:"
  echo "$(python3 --version 2>&1)"
  exit 1
fi

if ! [ -e venv ]; then
  python3 -m pip install virtualenv
  python3 -m virtualenv -p python3 venv
fi

venv/bin/pip install -e . --find-links https://download.pytorch.org/whl/torch_stable.html
venv/bin/pip check

# Python style checks and linting
venv/bin/black --check --diff src scripts || (
  echo ""
  echo "The code formatting check failed. To fix the formatting, run:"
  echo ""
  echo ""
  echo -e "\tvenv/bin/black src"
  echo ""
  echo ""
  exit 1
)

venv/bin/mypy --install-types --non-interactive src scripts
venv/bin/flake8 src scripts

echo "Done."
