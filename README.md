Welcome!  This internal repository contains all the assets for the CRFM/Mercury benchmarking project.

# Metrics

TODO.  Once this exists, we should move `src` to another directory.

# Access to language models

We provide a single unified entry point into accessing large language models
(e.g., GPT-3, Jurassic).  This provides both a web interface and an API.

To use the web interface, go to `http://crfm-models.stanford.edu`.

To use the API, see [demo.py](demo.py).

## Deployment (for maintainers)

Create `credentials.conf` to contain the API keys for the language models.

    openaiApiKey: ...
    ai21ApiKey: ...

To start a local server (go to `http://localhost:1959` to try it out):

    pip install -r requirements.txt
    python src/server.py

To update and start the public server (be careful!):

    rsync -arvz requirements.txt src credentials.conf crfm-models.stanford.edu:/home/benchmarking
    ssh crfm-models
    cd /home/benchmarking
    virtualenv venv
    venv/bin/pip install -r requirements.txt
    sudo venv/bin/python src/server.py -p 80

# Contributing

To contribute to this project, install the dev dependencies and git hook scripts:
`pip install -r requirements.txt && pre-commit install`.