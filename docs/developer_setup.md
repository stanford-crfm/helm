# Developer Setup

## Set up the Python virtual environment

First, create a Python virtual environment with Python version >= 3.8 and activate it.

Using [Virtualenv](https://docs.python.org/3/library/venv.html#creating-virtual-environments):

    # Create a virtual environment.
    # Only run this the first time.
    python3 -m pip install virtualenv
    python3 -m virtualenv -p python3 venv

    # Activate the virtual environment.
    source venv/bin/activate

Using [Anaconda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html):

    # Create a virtual environment.
    # Only run this the first time.
    conda create -n crfm-helm python=3.8 pip

    # Activate the virtual environment.
    conda activate crfm-helm

## Install dependencies

To install any dependencies:

    ./install-dev.sh

If you run into errors when installing dependencies, please create a new Python virtual environment using the previous instructions, and then try installing the dependencies again.

Optionally, install Git commit hooks:

    pre-commit install

## Run tests

First, follow the earlier instructions to activate the virtual environment.

To run all unit tests:

    python -m pytest

Append `-vv` to output the full diff and results:

    python -m pytest -vv

To run a specific file, simply specify the path:

    python -m pytest <path/to/file> -vv

## Executing helm commands with local modifications

The recommended way to execute `helm-run`, `helm-summarize`, `helm-server`, etc, with your local version of the repository is to do an editable install, using the following steps:

1. Activate your virtual environment.
1. Change directory to the repository root (contains setup.cfg).
1. Make sure you don't have an existing helm installation for that environmnt with `pip uninstall crfm-helm`
1. Run `pip install -e .`

Now calling `helm-run` while the environment is activated will read from your local source.

### Without installing

If you have a compelling reason not to do an editable install, you can execute commands by:

1. Change directory `src`
1. Execute the module you want with a command like: `python -m helm.benchmark.run`

## Run linter and type-checker

To run the linter and type-checker:

    ./pre-commit.sh

If you previously installed the Git commit hooks using `pre-commit install`, then the linter and type-checker will be run whenever you run `git commit`. To skip running the linter and type checker when making a commit, use the `--no-verify` flag with `git commit`.
