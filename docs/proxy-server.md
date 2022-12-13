# Proxy Access to Language Models

We provide a single unified entry point into accessing large language models
(e.g., GPT-3, Jurassic).  This provides both a web interface and a REST API.

## Using (for most people)

To use the web interface, go to https://crfm-models.stanford.edu.

To use the REST API, see [demo.py](https://github.com/stanford-crfm/benchmarking/blob/main/demo.py).

## Deploying locally

Create `prod_env/credentials.conf` to contain the API keys for any language
models you have access to.

    openaiApiKey: ...
    ai21ApiKey: ...

To start a local server (go to `http://localhost:1959` to try it out):

    venv/bin/crfm-proxy-server

When starting the server for the first time, the server will create an admin account 
with the API key: `root`.
If you're deploying the server to production, make sure to rotate the API key of the
default admin account.

### For macOS developers

Bypass the added security that restricts multithreading by running:

    OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES venv/bin/crfm-proxy-server
