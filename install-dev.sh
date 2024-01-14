#!/bin/bash

# This script installs pinned dependencies, then install CRFM-HELM in edit mode

set -e

# On Mac OS, skip installing pytorch with CUDA because CUDA is not supported
if [[ $OSTYPE != 'darwin'* ]]; then
  # Manually install pytorch to avoid pip getting killed: https://stackoverflow.com/a/54329850
  pip install --no-cache-dir --find-links https://download.pytorch.org/whl/torch_stable.html torch==2.0.1+cu118 torchvision==0.15.2+cu118
fi
# Install all pinned dependencies
pip install -r requirements.txt
# upgrade pip to install in edit mode without setup.py
pip install --upgrade pip
# Install HELM in edit mode
pip install -e .[all]
# Check dependencies
pip check
