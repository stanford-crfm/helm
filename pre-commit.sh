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

# On Mac OS, skip installing pytorch with CUDA because CUDA is not supported
if [[ $OSTYPE != 'darwin'* ]]; then
  # Manually install pytorch to avoid pip getting killed: https://stackoverflow.com/a/54329850
  pip install --no-cache-dir --find-links https://download.pytorch.org/whl/torch_stable.html torch==1.12.1+cu113 torchvision==0.13.1+cu113
fi
# Manually install protobuf to workaround issue: https://github.com/protocolbuffers/protobuf/issues/6550
pip install --no-binary=protobuf protobuf==3.20.2
# Install all pinned dependencies
pip install -r requirements-freeze.txt
pip install -e .
pip check

# Python style checks and linting
black --check --diff src scripts || (
  echo ""
  echo "The code formatting check failed. To fix the formatting, run:"
  echo ""
  echo ""
  echo -e "\tblack src scripts"
  echo ""
  echo ""
  exit 1
)

mypy --install-types --non-interactive src scripts
flake8 src scripts

echo "Done."
