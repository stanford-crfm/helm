#!/bin/bash

# This script runs the pre-commit using virtualenv for GitHub Actions.

set -e

if ! [ -e venv ]; then
  python3 -m pip install virtualenv
  python3 -m virtualenv -p python3 venv
fi

source venv/bin/activate

./pre-commit.sh
