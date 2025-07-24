# syntax=docker/dockerfile:1.5
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV PIP_ROOT_USER_ACTION=ignore

# Control which python version we are using
ARG PYTHON_VERSION=3.10

# Control the version of uv
ARG UV_VERSION=0.7.19

# Control the version of HELM (by default uses the current branch)
ARG HELM_GIT_HASH=HEAD

# ------------------------------------
# Step 1: Install System Prerequisites
# ------------------------------------
RUN <<EOF
#!/bin/bash
set -e
apt update -q
DEBIAN_FRONTEND=noninteractive apt install -q -y --no-install-recommends \
    curl \
    wget \
    git \
    ca-certificates \
    build-essential 
# Cleanup for smaller image sizes
apt clean
rm -rf /var/lib/apt/lists/*
EOF

# Set the shell to bash to auto-activate enviornments
SHELL ["/bin/bash", "-l", "-c"]

# ------------------
# Step 2: Install uv
# ------------------
# Here we take a few extra steps to pin to a verified version of the uv
# installer. This increases reproducibility and security against the main
# astral domain, but not against those linked in the main installer.
# The "normal" way to install the latest uv is:
# curl -LsSf https://astral.sh/uv/install.sh | bash
RUN <<EOF
#!/bin/bash
set -e
mkdir /bootstrap
cd /bootstrap
declare -A UV_INSTALL_KNOWN_HASHES=(
    ["0.7.20"]="3b7ca115ec2269966c22201b3a82a47227473bef2fe7066c62ea29603234f921"
    ["0.7.19"]="e636668977200d1733263a99d5ea66f39d4b463e324bb655522c8782d85a8861"
)
EXPECTED_SHA256="${UV_INSTALL_KNOWN_HASHES[${UV_VERSION}]}"
DOWNLOAD_PATH=uv-install-v${UV_VERSION}.sh
if [[ -z "$EXPECTED_SHA256" ]]; then
    echo "No hash known for UV_VERSION '$UV_VERSION'; no known hash. Aborting."
    exit 1
fi
curl -LsSf https://astral.sh/uv/$UV_VERSION/install.sh > $DOWNLOAD_PATH
echo "$EXPECTED_SHA256  $DOWNLOAD_PATH" | sha256sum --check
# Run the install script
bash /bootstrap/uv-install-v${UV_VERSION}.sh
# Cleanup for smaller images
rm -rf /root/.cache/
EOF


# ------------------------------------------
# Step 3: Setup a Python virtual environment
# ------------------------------------------
# This step mirrors a normal virtualenv development environment inside the
# container, which can prevent subtle issues due when running as root inside
# containers. 
RUN <<EOF
#!/bin/bash
export PATH="$HOME/.local/bin:$PATH"
# Use uv to install the requested python version and seed the venv
uv venv "/root/venv$PYTHON_VERSION" --python=$PYTHON_VERSION --seed
BASHRC_CONTENTS='
# setup a user-like environment, even though we are root
export HOME="/root"
export PATH="$HOME/.local/bin:$PATH"
# Auto-activate the venv on login
source /root/venv'$PYTHON_VERSION'/bin/activate
'
# It is important to add the content to both so 
# subsequent run commands use the the context we setup here.
echo "$BASHRC_CONTENTS" >> $HOME/.bashrc
echo "$BASHRC_CONTENTS" >> $HOME/.profile
EOF


RUN mkdir -p /root/code/helm

# ---------------------------------
# Step 4: Checkout and install HELM
# ---------------------------------
# Based on the state of the repo this copies the host .git data over and then
# checks out the extact version of HELM requested by HELM_GIT_HASH. It then
# performs a basic install of helm into the virtual environment.
COPY .git /root/code/helm/.git
RUN <<EOF
set -e

cd  /root/code/helm

# Checkout the requested branch 
git checkout "$HELM_GIT_HASH"
git reset --hard "$HELM_GIT_HASH"

# TODO: cleanup once we determine the best way to 
# install the HELM package for reproducibility. 

# First install pinned requirements for reproducibility
uv pip install -r requirements.txt

# 
# Install helm in developement mode
# uv pip install -e .

uv pip install -e .[all,dev] 
#--resolution lowest-direct

# Cleanup for smaller cache
rm -rf /root/.cache/
EOF


# -----------------------------------
# Step 5: Ensure venv auto-activation
# -----------------------------------
# This final steps ensures that commands the user provides to docker run
# will always run in in the context of the virtual environment. 
RUN  <<EOF
#!/bin/bash
set -e
# write the entrypoint script
echo '#!/bin/bash
set -e
# Build the escaped command string
cmd=""
for arg in "$@"; do
  # Use printf %q to properly escape each argument for bash
  cmd+=$(printf "%q " "$arg")
done
# Remove trailing space
cmd=${cmd% }
exec bash -lc "$cmd"
' > entrypoint.sh
chmod +x /entrypoint.sh
EOF

# Set the entrypoint to our script that activates the virtual enviornment first
ENTRYPOINT ["/entrypoint.sh"]

# Set the default workdir to the helm code repo
WORKDIR /root/code/helm

# ---------------------------------------------------------------
# End of dockerfile logic. The following lines are documentation.
# ---------------------------------------------------------------

# HACK
RUN <<EOF
#!/bin/bash
set -e
apt update -q
DEBIAN_FRONTEND=noninteractive apt install -q -y --no-install-recommends \
    wget \
# Cleanup for smaller image sizes
apt clean
rm -rf /var/lib/apt/lists/*
EOF

################
### __DOCS__ ###
################
RUN <<EOF
echo 'HEREDOC:
# https://www.docker.com/blog/introduction-to-heredocs-in-dockerfiles/

# The following are instructions to build and test this docker image

# cd into a local clone of the helm repo
cd ~/code/helm/

# Determine which helm version to use
HELM_GIT_HASH=$(git rev-parse --short=12 HEAD)

# Build HELM in a reproducible way.
DOCKER_BUILDKIT=1 docker build --progress=plain \
    -t helm:$HELM_GIT_HASH-uv0.7.29-python3.10 \
    --build-arg PYTHON_VERSION=3.10 \
    --build-arg UV_VERSION=0.7.19 \
    --build-arg HELM_GIT_HASH=$HELM_GIT_HASH \
    -f ./dockerfiles/helm.dockerfile .

# Add latest tags for convinience
docker tag helm:$HELM_GIT_HASH-uv0.7.29-python3.10 helm:latest-uv0.7.29-python3.10
docker tag helm:$HELM_GIT_HASH-uv0.7.29-python3.10 helm:latest

# Verify that GPUs are visible and that each helm command works
docker run --gpus=all -it helm:latest nvidia-smi
docker run --gpus=all -it helm:latest helm-run --help
docker run --gpus=all -it helm:latest helm-summarize --help
docker run --gpus=all -it helm:latest helm-server --help

# Run a small end-to-end benchmark.

# Create a shared directory so results persist outside of the container.
# We will then run commands in the container using that shared path as the
# working directory.
mkdir -p ./shared_directory

# Run benchmarks
docker run --rm --gpus=all \
    -v $PWD/shared_directory:/mnt/shared_directory \
    --workdir /mnt/shared_directory \
    -it helm:latest \
    helm-run --run-entries mmlu:subject=philosophy,model=openai/gpt2 --suite my-suite --max-eval-instances 10

# Summarize benchmark results
docker run --rm --gpus=all \
    -v $PWD/shared_directory:/mnt/shared_directory \
    --workdir /mnt/shared_directory \
    -it helm:latest \
    helm-summarize --suite my-suite

# Start a web server to display benchmark results
docker run --rm --gpus=all \
    -v $PWD/shared_directory:/mnt/shared_directory \
    --workdir /mnt/shared_directory \
    -p 8000:8000 \
    -it helm:latest \
    helm-server --suite my-suite

' > /dev/null

EOF
