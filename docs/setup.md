# Setup

To install any dependencies (creates into `venv`):

    ./pre-commit.sh

Note that you will need to make sure that python3 is at least version 3.8 (the default version in the NLP cluster is 3.6, so you will need to create a custom conda environment). To check your python version:
    
    python3 --version

Optionally, install git commit hooks:

    pre-commit install

## Tests

First activate the virtualenv:

    source venv/bin/activate

To run all unit tests:

    python -m pytest

Append `-vv` to output the full diff and results:

    python -m pytest -vv

To run a specific file, simply specify the path:

    python -m pytest <path/to/file> -vv
