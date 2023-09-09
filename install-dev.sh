#!/bin/bash

# This script installs pinned dependencies, then install CRFM-HELM in edit mode

set -e

# On Mac OS, skip installing pytorch with CUDA because CUDA is not supported
if [[ $OSTYPE != 'darwin'* ]]; then
  # Manually install pytorch to avoid pip getting killed: https://stackoverflow.com/a/54329850
  pip install --no-cache-dir --find-links https://download.pytorch.org/whl/torch_stable.html torch==1.12.1+cu113 torchvision==0.13.1+cu113
fi
# Manually install protobuf to workaround issue: https://github.com/protocolbuffers/protobuf/issues/6550
pip install --no-binary=protobuf protobuf==3.20.2
# Install all pinned dependencies
pip install -r requirements-freeze.txt
# Install HELM in edit mode
pip install -e .[all]
# Check dependencies
pip check
