Welcome!  This internal repository contains all the assets for the CRFM/Mercury
benchmarking project.  There are two related parts:

1. Proxy (`src/proxy`): provides a unified way to access major language models.
2. Benchmarking (see `src/benchmark`): evaluates such language models.

# Setup

To install any dependencies (into `venv`):

    ./pre-commit.sh

# Proxy access to language models

We provide a single unified entry point into accessing large language models
(e.g., GPT-3, Jurassic).  This provides both a web interface and a REST API.

## Using (for most people)

To use the web interface, go to `http://crfm-models.stanford.edu` (TODO: update
this to https).

To use the REST API, see [demo.py](demo.py).  (TODO: provide some Python classes.)

## Deploying locally (for developers)

Create `prod_env/credentials.conf` to contain the API keys for any language
models you have access to.

    openaiApiKey: ...
    ai21ApiKey: ...

To start a local server (go to `http://localhost:1959` to try it out):

    venv/bin/proxy-server

## Deploying to production (for maintainers)

The production version of the proxy is running on `crfm-models.stanford.edu`;
you need to get permission to get ssh access.

### One-time setup

This is done, but just for the record:

    laptop:$ ssh crfm-models.stanford.edu
    crfm-models:$ cd /home
    crfm-models:$ git clone git@github.com:stanford-crfm/benchmarking
    crfm-models:$ cd benchmarking
    crfm-models:$ mkdir prod_env
    laptop:$ rsync -arvz prod_env/credentials.conf crfm-models.stanford.edu:/home/benchmarking/prod_env

### Every time we need to deploy:

Update the code:

    laptop:$ ssh crfm-models.stanford.edu
    crfm-models:$ cd /home/benchmarking
    crfm-models:$ git pull
    crfm-models:$ ./pre-commit.sh

If everything looks okay:

    # Switch into the screen session
    crfm-models:$ gos bench

    # Hit ctrl-c to kill the existing process

    sudo venv/bin/proxy-server -p 80  # TODO: replace this with https

Double check that the web site still works.

# Benchmarking

## Code structure

Here's a birds-eye view of how the benchmarking process interacts with the main
classes (see `benchmark`):

- A `Scenario` (given by a `ScenarioSpec`) specifies a task and a data
  distribution.  It specifies a set of `Instance`s, where each `Instance` has
  an input (e.g., question) and a set of `Reference` outputs (e.g., multiple
  choice answers).

- An `Adapter` (given by an `AdaptationSpec`) takes a `Scenario` and
  adapts it to a set of `Request`s to the API (e.g., the model, temperature,
  number of in-context training examples).  Formally, the output
  is a `ScenarioState` containing a set of `RequestState`s, where each
  `RequestState` consists of a `Request` and any metadata used to track the
  role of this `Request` (e.g., the relevant `Instance` and `Reference`).

- An `Executor` (given by an `ExecutionSpec`) executes each `Request` in the
  `RequestState` to produce a `RequestResult` for each one; everything is
  encapsulated in a `ScenarioState`.

- A `Metric` (given by a `MetricSpec`) takes a `ScenarioState` containing
  `RequestResults`s and produces a set of `Stat`s (e.g., accuracy, accuracy@5,
  toxicity, bias, etc.).

- A `Runner` is the top-level controller that runs the above steps and is
  driven by a set of `RunSpec`s.

There are three types of classes:

- Specifications (e.g., `AdaptationSpec`, `ExecutionSpec`, `RunSpec`):
  specified manually by the user.  Note that `Scenario` and `Metric` are
  subclassed, so they are constructed by `ObjectSpec`, which specifies the
  subclass name and a free-form dictionary of arguments.
- States (e.g., `Instance`, `ScenarioState`, `Request`, `RequestResult`): these
  are automatically generated and can be serialized.
- Controllers (e.g., `Scenario`, `Adapter`, `Executor`, `Metric, `Runner`):
  these have the bulk of the code and should not be serialized.

## Running the benchmark

Run the benchmark:

    venv/bin/benchmark-run

You can also run the benchmark using a local proxy, in which case you have to
first start a local server (see instructions above for more details).

# Contributing

To contribute to this project, install the dependencies and git hook scripts:
  
    ./pre-commit.sh && pre-commit install

To run unit tests:

    python -m pytest
