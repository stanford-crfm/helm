# Setup

To install any dependencies (creates into `venv`):

    ./pre-commit.sh

If you get the following error during installation, please delete the `venv` folder and re-run the above command:

    Installing build dependencies ... error
    error: subprocess-exited-with-error

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
