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
  # DALLE mini requires jax install
  pip install jax==0.3.25 jaxlib==0.3.25+cuda11.cudnn805 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
fi

# Manually install protobuf to workaround issue: https://github.com/protocolbuffers/protobuf/issues/6550
pip install --no-binary=protobuf protobuf==3.20.1

# For CogView2, manually install apex and Image-Local-Attention. NOTE: need to run this on a GPU machine
if (nvcc --version > /dev/null 2>&1); then
    ROOT=`exec pwd`
    mkdir -p tmp && chmod -R 777 tmp && rm -r tmp
    mkdir -p tmp && cd tmp && git clone https://github.com/Sleepychord/Image-Local-Attention && cd Image-Local-Attention && git checkout 43fee310cb1c6f64fb0ed77404ba3b01fa586026 && python setup.py install
    cd $ROOT
    mkdir -p tmp && cd tmp && git clone https://github.com/michiyasunaga/apex && cd apex && git checkout 9395ba2aab3c05e0e36ef0b7fe48d42de9f10bcf && pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
    cd $ROOT
else
    ROOT=`exec pwd`
    mkdir -p tmp && chmod -R 777 tmp && rm -r tmp
    mkdir -p tmp && cd tmp && git clone https://github.com/michiyasunaga/apex && cd apex && git checkout 9395ba2aab3c05e0e36ef0b7fe48d42de9f10bcf && pip install -v --disable-pip-version-check --no-cache-dir ./
    cd $ROOT
fi


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

mkdir -p .mypy_cache
mypy --install-types --non-interactive src scripts
flake8 src scripts

echo "Done."
