Welcome!  This internal repository contains all the assets for the CRFM/Mercury benchmarking project.

# Benchmarking

Here's a birds-eye view of how the benchmarking process interacts with the main classes:

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

Start a server locally (see below).

    python src/run_benchmark.py


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
  
    pip install -r requirements.txt && pre-commit install

To run unit tests:

    python -m pytest
