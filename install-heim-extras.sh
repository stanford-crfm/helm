#!/bin/bash

# Extra dependencies for HEIM when evaluating the following:
# Models: craiyon/dalle-mini, craiyon/dalle-mega, thudm/cogview2
# Scenarios: detection with the `DetectionMetric`

# This script fails when any of its commands fail.
set -e

# For DALLE-mini/mega, install the following dependencies.
# On Mac OS, skip installing pytorch with CUDA because CUDA is not supported
if [[ $OSTYPE != 'darwin'* ]]; then
  # Manually install pytorch to avoid pip getting killed: https://stackoverflow.com/a/54329850
  pip install --no-cache-dir --find-links https://download.pytorch.org/whl/torch_stable.html torch==1.12.1+cu113 torchvision==0.13.1+cu113

  # DALLE mini requires jax install
  pip install jax==0.3.25 jaxlib==0.3.25+cuda11.cudnn805 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
fi

# For CogView2, manually install apex and Image-Local-Attention. NOTE: need to run this on a GPU machine
echo "Installing CogView2 dependencies..."
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

# For Detectron2. Following https://detectron2.readthedocs.io/en/latest/tutorials/install.html
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

echo "Done."
