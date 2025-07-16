#!/bin/bash

# This script installs pinned dependencies, then install CRFM-HELM in edit mode

set -e

pip install --upgrade setuptools
# On Mac OS, skip installing pytorch with CUDA because CUDA is not supported
if [[ $OSTYPE != 'darwin'* ]]; then
  # Manually install pytorch with `--no-cache-dir` to avoid pip getting killed: https://stackoverflow.com/a/54329850
  pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir
fi
# Install all pinned dependencies
pip install -r requirements.txt
# Install HELM in edit mode
pip install -e .[all,dev]
