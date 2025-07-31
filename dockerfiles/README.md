# The HELM Docker Image

This guide outlines how to build, tag, and test the HELM Docker image. Start by
navigating to a local clone of the HELM repository, and then navigating to it:

```bash
git clone https://github.com/stanford-crfm/helm.git ~/code/helm
cd ~/code/helm/
```

### Building the Image

To ensure reproducibility, specify the versions of HELM, `uv`, and Python via
build arguments. This guarantees that builds are consistent across machines and
environments.

Run docker build with build args to specify exactly which variant of HELM we
are building and in which environment it will run. This ensures
reproducibility.

```bash
# Determine version of helm, uv, and python to use
export HELM_GIT_HASH=$(git rev-parse --short=12 HEAD)
export UV_VERSION=0.8.4
export PYTHON_VERSION=3.10

# Build the image with version-specific tags
DOCKER_BUILDKIT=1 docker build --progress=plain \
    -t helm:${HELM_GIT_HASH}-uv${UV_VERSION}-python${PYTHON_VERSION} \
    --build-arg PYTHON_VERSION=$PYTHON_VERSION \
    --build-arg UV_VERSION=$UV_VERSION \
    --build-arg HELM_GIT_HASH=$HELM_GIT_HASH \
    -f ./dockerfiles/helm.dockerfile .
```


After building, tag the image with shorter aliases for convenience:

```bash
# Add concise tags for easier reuse
docker tag helm:${HELM_GIT_HASH}-uv${UV_VERSION}-python${PYTHON_VERSION} helm:latest-uv${UV_VERSION}-python${PYTHON_VERSION}
docker tag helm:${HELM_GIT_HASH}-uv${UV_VERSION}-python${PYTHON_VERSION} helm:latest-python${PYTHON_VERSION}
docker tag helm:${HELM_GIT_HASH}-uv${UV_VERSION}-python${PYTHON_VERSION} helm:latest
```


### Smoke Testing the Image

Perform basic sanity checks to ensure:

* The GPU is accessible (if available)

```bash
docker run --gpus=all -it helm:latest nvidia-smi
```

* All CLI entry points are wired up correctly

```bash
docker run --gpus=all -it helm:latest helm-run --help
docker run --gpus=all -it helm:latest helm-summarize --help
docker run --gpus=all -it helm:latest helm-server --help
```

### Unit Tests

This Docker image is designed as a development container, so you can also
verify the HELM installation by running the unit tests inside the image.


```bash
docker run --rm --gpus=all \
    -it helm:latest \
    pytest
```

### End-to-End Benchmark Test

For a more robust validation—and to demonstrate typical usage—you can run a
small benchmark and persist results using a shared host volume.

Create a shared directory on the host for output persistence

```bash
mkdir -p ./shared_directory
```

Run a benchmark:

```bash
docker run --rm --gpus=all \
    -v $PWD/shared_directory:/mnt/shared_directory \
    --workdir /mnt/shared_directory \
    -it helm:latest \
    helm-run --run-entries mmlu:subject=philosophy,model=openai/gpt2 --suite my-suite --max-eval-instances 10
```

Summarize the results:

```bash
docker run --rm --gpus=all \
    -v $PWD/shared_directory:/mnt/shared_directory \
    --workdir /mnt/shared_directory \
    -it helm:latest \
    helm-summarize --suite my-suite
```

Start a web server to view the results:

```bash
docker run --rm --gpus=all \
    -v $PWD/shared_directory:/mnt/shared_directory \
    --workdir /mnt/shared_directory \
    -p 8000:8000 \
    -it helm:latest \
    helm-server --suite my-suite
```
