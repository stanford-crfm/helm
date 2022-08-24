#!/bin/bash

# This script fails when any of its commands fail.
set -e

if ! [ -e venv ]; then
  python3 -m pip install virtualenv
  python3 -m virtualenv -p python3 venv
fi

venv/bin/pip install -e . --find-links https://download.pytorch.org/whl/torch_stable.html
# Issue + workaround described here: https://github.com/protocolbuffers/protobuf/issues/6550
venv/bin/pip uninstall -y protobuf && venv/bin/pip install --no-binary=protobuf protobuf
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

venv/bin/mypy src scripts
venv/bin/flake8 src scripts

echo "Done."
