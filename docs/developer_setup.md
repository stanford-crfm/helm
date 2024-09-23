# Developer Setup

## Check your system Python version

Check your system verison of Python by running:

```bash
python --version
```

If your version of Python is older than 3.9, you _must_ use either **Conda** or **pyenv** to install a version of Python >=3.9 when setting up your virtual environment.

## Set up the Python virtual environment

First, create a Python virtual environment with Python version >= 3.9 and activate it.

Using [**Virtualenv**](https://docs.python.org/3/library/venv.html#creating-virtual-environments) (*requires* system Python version >=3.9):

```bash
# Create a virtual environment.
# Only run this the first time.
python3 -m pip install virtualenv
python3 -m virtualenv -p python3 venv

# Activate the virtual environment.
# Run this every time you open your shell.
source venv/bin/activate
```

Using [**Conda**](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html):

```bash
# Create a virtual environment.
# Only run this the first time.
conda create -n crfm-helm python=3.10 pip

# Activate the virtual environment.
# Run this every time you open your shell.
conda activate crfm-helm
```

Using [**pyenv**](https://github.com/pyenv/pyenv) and [**pyenv-virtualenv**](https://github.com/pyenv/pyenv-virtualenv):

```bash
# Create a virtual environment.
# Only run this the first time.
pyenv virtualenv 3.10 crfm-helm

# Activate the virtual environment.
# Run this every time you open your shell.
pyenv activate crfm-helm
```

## Install Python dependencies

To install any dependencies:

```bash
pip install --force-reinstall -e .[dev]
```

## Run Python tests

Currently, running all the unit tests takes about 10 minutes. To run all unit tests:

```bash
python -m pytest
```

Append `-vv` to output the full diff and results:

```bash
python -m pytest -vv
```

When modifying the Python code, you usually want to only run certain relevant tests. To run a specific test file, specify the file path as follows:

```bash
python -m pytest path/to/test_file.py -vv
```

## Run linter and type-checker

You should always ensure that your code is linted and type-checked before creating a pull request. This is typically enforced by our git pre-commit hooks. Install the pre-commit hooks by running:

```bash
pre-commit install
```

This will automatically run the linter and type-checker whenever you run `git push` to push a branch. To skip running the linter and type checker when pushing a branch, use the `--no-verify` flag with `git push`.

To run the linter and type-checker manually:

```bash
./pre-commit.sh
```

Alternatively, you can run only the linter or only the type checker separately:

```bash
# Linters
black src scripts
flake8 src scripts

# Type checker
mypy src scripts
```
