# Developer Setup

## Check your system Python version

Check your system verison of Python by running:

```bash
python --version
```

If your version of Python is older than 3.10, you _must_ use either **Conda** or **pyenv** to install a version of Python >=3.10 when setting up your virtual environment.

## Set up the Python virtual environment

First, create a Python virtual environment with Python version >= 3.10 and activate it.

Using [**Virtualenv**](https://docs.python.org/3/library/venv.html#creating-virtual-environments) (*requires* system Python version >=3.10):

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

## Executing helm commands with local modifications

The recommended way to execute `helm-run`, `helm-summarize`, `helm-server`, etc, with your local version of the repository is to do an editable install, using the following steps:

1. Activate your virtual environment.
1. Change directory to the repository root (contains pyproject.toml).
1. Make sure you don't have an existing helm installation for that environment with `pip uninstall crfm-helm`
1. Run `pip install -e .`

Now calling `helm-run` while the environment is activated will read from your local source.

### Without installing

If you have a compelling reason not to do an editable install, you can execute commands by:

1. Change directory to `src`
1. Execute the module you want with a command like: `python -m helm.benchmark.run`

## Checking in code

The HELM repository does not allow direct modifications of the main branch. Instead, developers create a Pull Request which must then be approved by a different person before merging into main. Here is an example workflow:

1. `git checkout main` to start from the main branch.
1. `git pull origin main` to get up to date.
1. Make whatever changes you'll like to group into a single review.
1. Run tests.
1. Make a new branch with `git checkout -b <your-handle>/<change-identifier`. For example, `yifanmai/fix-optional-suggestions`.
1. If you did NOT install the precommit, run the linter and type checker with `./pre-commit.sh`
1. `git commit -a` to commit all you changes. If you want to ignore precommit warnings,  you can add `--no-verify`.
1. `git push origin <your-handle>/<change-identifier>` to upload to github.
1. Loading any HELM github page should now prompt you about creating a new pull request. If not, you can also find your branch on [the branches page](https://github.com/stanford-crfm/helm/branches) to create one.
1. Update the title and description as necessary, then create the pull request.
1. Once the reviewer is satisfied, they can approve and either of you can then `Squash and Merge` the branch into main.
