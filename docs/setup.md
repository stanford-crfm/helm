# Setup

To install any dependencies (creates into `venv`):

    ./pre-commit.sh

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
