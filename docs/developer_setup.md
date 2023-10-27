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
