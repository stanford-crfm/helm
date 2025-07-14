# syntax=docker/dockerfile:1.5
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV PIP_ROOT_USER_ACTION=ignore

# Control which python version we are using
ARG PYTHON_VERSION=3.10

ARG UV_VERSION=0.7.19

# Install Prerequisites base tools, system Python (only needed for bootstrapping)
RUN <<EOF
#!/bin/bash
set -e
apt update -q
DEBIAN_FRONTEND=noninteractive apt install -q -y --no-install-recommends \
    curl \
    wget \
    git \
    ca-certificates \
    python3 \
    python3-venv \
    python3-pip \
    build-essential 
apt clean
rm -rf /var/lib/apt/lists/*
EOF

# Set the shell to bash to auto-activate enviornments
SHELL ["/bin/bash", "-l", "-c"]

# Install uv
#
# NOTE: We pin to a specific version of the astral install script to be more
# reproducible and secure against compromised domains.
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
echo "EXPECTED_SHA256=$EXPECTED_SHA256"
echo "DOWNLOAD_PATH=$DOWNLOAD_PATH"
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


# Install uv
RUN <<EOF
#!/bin/bash
export PATH="$HOME/.cargo/bin:$PATH"
export PATH="$HOME/.local/bin:$PATH"

# Use uv to install Python 3.13 and seed venv
uv venv "/root/venv$PYTHON_VERSION" --python=$PYTHON_VERSION --seed

echo "Write bashrc"
BASHRC_CONTENTS='
# setup a user-like environment, even though we are root
export HOME="/root"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
# Auto-activate the venv on login
source /root/venv'$PYTHON_VERSION'/bin/activate
'
# It is important to add the content to both so 
# subsequent run commands use the the context we setup here.
echo "$BASHRC_CONTENTS" >> $HOME/.bashrc
echo "$BASHRC_CONTENTS" >> $HOME/.profile
EOF

ENTRYPOINT ["/bin/bash", "-lc"]


RUN mkdir -p /root/code/helm

# Set the default workdir to the helm code repo
WORKDIR /root/code/helm

ARG HELM_GIT_HASH=HEAD

# Copy the .git data over for a fresh install in docker
COPY .git /root/code/helm/.git

RUN <<EOF
set -e

# Checkout the requested branch 
git checkout "$HELM_GIT_HASH"
git reset --hard "$HELM_GIT_HASH"

# Install helm in developement mode
uv pip install -e .

# Cleanup for smaller cache
rm -rf /root/.cache/
EOF




# Create an entrypoint that will run a command in the context of the bash environment
RUN  <<EOF
#!/bin/bash
set -e
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
' > /entrypoint.sh
chmod +x /entrypoint.sh
cat /entrypoint.sh
EOF

ENTRYPOINT ["/entrypoint.sh"]


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

DOCKER_BUILDKIT=1 docker build --progress=plain \
    -t helm:$HELM_GIT_HASH-uv0.7.29-python3.10 \
    --build-arg PYTHON_VERSION=3.10 \
    --build-arg UV_VERSION=0.7.19 \
    --build-arg HELM_GIT_HASH=$HELM_GIT_HASH \
    -f ./dockerfiles/helm.dockerfile .


# Get a shell where you can run any commands
docker run --gpus=all -it helm:$HELM_GIT_HASH-uv0.7.29-python3.10 bash


# Test the build

echo "
# Run benchmark
helm-run --run-entries mmlu:subject=philosophy,model=openai/gpt2 --suite my-suite --max-eval-instances 10

# Summarize benchmark results
helm-summarize --suite my-suite

# Start a web server to display benchmark results
helm-server --suite my-suite
" > run_quickstart.sh
chmod +x run_quickstart.sh

docker run \
    --gpus=all \
    -p 8000:8000 \
    -v $PWD:/mnt/host \
    -w /mnt/host \
    -it helm:$HELM_GIT_HASH-uv0.7.29-python3.10 \
    ./run_quickstart.sh

' > /dev/null
EOF
