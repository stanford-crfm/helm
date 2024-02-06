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
  pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir

  # DALLE mini requires jax install
  pip install jax==0.3.25 jaxlib==0.3.25+cuda11.cudnn805 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
fi

# For CogView2, manually install apex and Image-Local-Attention. NOTE: need to run this on a GPU machine
echo "Installing CogView2 dependencies..."
pip install localAttention@git+https://github.com/Sleepychord/Image-Local-Attention.git@43fee310cb1c6f64fb0ed77404ba3b01fa586026
pip install --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" apex@git+https://github.com/michiyasunaga/apex.git@9395ba2aab3c05e0e36ef0b7fe48d42de9f10bcf

# For Detectron2. Following https://detectron2.readthedocs.io/en/latest/tutorials/install.html
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

echo "Done."
