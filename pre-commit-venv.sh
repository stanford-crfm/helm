#!/bin/bash

# This script runs the pre-commit using virtualenv for GitHub Actions.

set -e

# Set up virtualenv
if ! [ -e venv ]; then
  python3 -m pip install virtualenv
  python3 -m virtualenv -p python3 venv
fi

source venv/bin/activate

# Install pinned dependencies, then install CRFM-HELM in edit mode
./install-dev.sh

# Run the main pre-commit script
./pre-commit.sh
