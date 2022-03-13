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

To use the web interface, go to https://crfm-models.stanford.edu.

To use the REST API, see [demo.py](demo.py).

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
    crfm-models:$ echo '{"api_key": "crfm"}' > prod_env/accounts.jsonl
    laptop:$ rsync -arvz prod_env/credentials.conf crfm-models.stanford.edu:/home/benchmarking/prod_env

#### Perspective API

We use Google's [Perspective API](https://www.perspectiveapi.com) to calculate the toxicity of completions.
To send requests to PerspectiveAPI, we need to generate an API key from GCP. Follow the
[Get Started guide](https://developers.perspectiveapi.com/s/docs-get-started)
to request the service and the [Enable the API guide](https://developers.perspectiveapi.com/s/docs-enable-the-api)
to generate the API key. Once you have a valid API key, add an entry to `credentials.conf`:

```
perspectiveApiKey: <Generated API key>
```

By default, Perspective API allows only 1 query per second. Fill out this
[form](https://developers.perspectiveapi.com/s/request-quota-increase) to increase the request quota.

The [current API key](https://console.cloud.google.com/apis/api/commentanalyzer.googleapis.com/overview?authuser=1&project=hai-gcp-models)
we are using in production was created with the `hai-gcp-models` account and allows 100 queries per second.
**The API key expires on 4/15/2022.**

#### SSL

The SSL certificate, CSR and private key for crfm-models.stanford.edu is stored at `/home/ssl`.
**The current SSL certificate expires on 12/30/2022.**

To renew the SSL certificate, follow these steps:

1. Fill out this [form](https://certificate.stanford.edu/cert-request):
    1. Log on with your SUNet ID. You must be an admin in order to submit a request.
    1. For `Server Name`, put `crfm-models.stanford.edu`.
    1. For `Server type`, select `OTHER`.
    1. For `Contact group/mailman address`, enter your Stanford email address.
    1. Under `Copy and paste your CSR`, paste the content of `/home/ssl/public.csr`.
    1. Leave the optional fields blank and click `Submit`.
    1. You should receive your certificate by email within 2 business days.
2. Once you receive the SSL cert, concatenate the contents of `X509 Certificate only, Base64 encoded`
   with the contents of `X509 Intermediates/root only Reverse, Base64 encoded`
   and place it at path `/home/ssl/crfm-models.crt`. `crfm-models.crt` should look something like this:

   ```text
    -----BEGIN CERTIFICATE-----
    (Your Primary SSL certificate: .crt)
    -----END CERTIFICATE-----
    -----BEGIN CERTIFICATE-----
    (Your Intermediate certificate: reversed.crt)
    -----END CERTIFICATE-----
   ```
3. Restart the server.
4. Open the [website](https://crfm-models.stanford.edu) in a browser and verify the connection is secure.

##### Misplaced private key or CSR

If, for whatever reason, the private key or CSR is misplaced, generate new ones by running:

`sudo openssl req -new -nodes -newkey rsa:2048 -keyout private.key -out public.csr`

and fill out the form:

```text
Country Name (2 letter code) [AU]:US
State or Province Name (full name) [Some-State]:California
Locality Name (eg, city) []:Stanford
Organization Name (eg, company) [Internet Widgits Pty Ltd]:Stanford University
Organizational Unit Name (eg, section) []:CRFM
Common Name (e.g. server FQDN or YOUR name) []:crfm-models.stanford.edu
Email Address []:

Please enter the following 'extra' attributes
to be sent with your certificate request
A challenge password []:
An optional company name []:
```

Then, follow the steps above to request for a new SSL certificate.

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

    sudo venv/bin/proxy-server -p 443 --ssl-key-file /home/ssl/private.key --ssl-cert-file /home/ssl/crfm-models.crt

Double check that the [website](https://crfm-models.stanford.edu) still works.

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

- Specifications (e.g., `AdapterSpec`, `ExecutionSpec`, `RunSpec`):
  specified manually by the user.  Note that `Scenario` and `Metric` are
  subclassed, so they are constructed by `ObjectSpec`, which specifies the
  subclass name and a free-form dictionary of arguments.
- States (e.g., `Instance`, `ScenarioState`, `Request`, `RequestResult`): these
  are automatically generated and can be serialized.
- Controllers (e.g., `Scenario`, `Adapter`, `Executor`, `Metric`, `Runner`):
  these have the bulk of the code and should not be serialized.

## Data Augmentations

To apply data augmentation, create a `DataAugmenterSpec` with a list of 
`PerturbationSpec`s and pass it into `AdapterSpec`. The following is an
example:

```python
    data_augmenter_spec = DataAugmenterSpec(
        perturbation_specs=[
            PerturbationSpec(
                class_name="benchmark.augmentations.perturbation.ExtraSpacePerturbation", 
                args={"num_spaces": 5},
            )
        ],
        should_perturb_references=False,
        should_augment_train_instances=False,
        should_include_original_train=False,
        should_augment_eval_instances=True,
        should_include_original_eval=True,
    )
    adapter_spec = AdapterSpec(
        ...
        data_augmenter_spec=data_augmenter_spec
    )
```

In the example above, the `Adapter` will augment the set of evaluation instances by perturbing
the original set of instances with the `ExtraSpacePerturbation`, where spaces in the text are 
replaced with `num_spaces` number of spaces. 

We currently only support applying a single perturbation to an instance instead of chaining
multiple perturbations and applying it onto a single instance.

### Adding a new perturbation

To add a new perturbation to the framework, simply create a new class in `perturbation.py` that
extends the abstract class `Perturbation` and implement the `perturb` method which takes in
text and outputs the perturbed text. Add a test for the new perturbation in `test_perturbation.py`.


## Interaction mode

The following is an overview of the code for interaction mode and instructions on supporting new types of interactions:

- In `src/benchmark/interaction`, add a new file for the new type of interaction (see the example at 
  `src/benchmark/interaction/simple_interactions`. In this file, define two new classes: 
    - The first class should inherit `InteractionOutcome`. `InteractionOutcome` represents the outcome of an 
      interaction. Define new fields here necessary to analyze and compute metrics.
    - The second class should inherit `InteractionsProcessor`. Add an `__init__` method and override the `process` 
      method, which defines how to process the outputs of the interactions and generate a list of `InteractionOutcome`s. 
      For example, for a chatbot that generates JSON files of the conversations users had with it, in `process`, read 
      the conversations from the JSON  files and generate a list of objects that represent the outcomes of the 
      interactions. 
- When defining a new `RunSpec` in `src/benchmark/run_specs.py`, create a new `AdapterSpec` and pass in 
  `InteractionModeSpec` with `interaction_mode=True` and an `InteractionsProcessorSpec` which points to the new class 
  created above. Also, pass in an empty scenario (use the `get_empty_scenario_spec` function), as we do not process 
  `Instance`s during interaction mode. 
- When `interaction_mode` is set to `True`,  the `Adapter` will skip generating `RequestState`s from `Instance`s and 
  generate a `ScenarioState` with `interaction_outcomes` set to the list of `InteractionOutcome`s  generated by 
  `InteractionsProcessor.process`.
- Define a new `Metric` to analyze the `InteractionOutcome`s in `ScenarioState` and compute their metrics.


## Running the benchmark

Examples of running the benchmark:

    venv/bin/benchmark-run
    venv/bin/benchmark-run -r mmlu:subject=philosophy
    venv/bin/benchmark-run -r lpm:difficulty=easy
    venv/bin/benchmark-run -r twitter_aae:demographic=aa
    venv/bin/benchmark-run -r copyright:pilot_study=true
    venv/bin/benchmark-run -r the_pile:subset=OpenSubtitles
    venv/bin/benchmark-run -r wiki:subject=P31
    venv/bin/benchmark-run -r raft:subset=ade_corpus_v2
    venv/bin/benchmark-run -r natural_qa:mode=closedbook
    venv/bin/benchmark-run -r quac

You can also run the benchmark using a local proxy, in which case you have to
first start a local server (see instructions above for more details).

### To estimate token usage

To estimate token usage without making any requests, append the `--dry-run` option:

    venv/bin/benchmark-run -r <RunSpec to estimate token usage> --dry-run

For example, running `venv/bin/benchmark-run -r real_toxicity_prompts --dry-run` outputs:

```text
  Stats {
    openai/davinci_estimated_number_of_tokens[min=505.000, mean=514.957, max=536.000, sum=514957.000 (1000)]
  }
```

where `sum` indicates the estimated total number of tokens used for the specific `RunSpec`.

For the OpenAI models, we use a
[GPT-2 Tokenizer](https://github.com/stanford-crfm/benchmarking/blob/master/src/proxy/tokenizer/openai_token_counter.py#L12)
to estimate the token usage. The tokenizer will be downloaded and cached when running a dry run.

## Final benchmarking (Infrastructure team only)

1. Running all the `RunSpec`s can take a long time, so use SCDT: `ssh scdt`.
1. Create a screen session: `screen -S benchmarking`.
1. Go to the source code directory: `cd /u/scr/nlp/crfm/benchmarking/benchmarking`.
   We have 700 GB of disk space total on `/u/scr/nlp/crfm`.
1. Pull the latest changes: `git pull`.
1. Run `venv/bin/benchmark-present -s /u/apache/htdocs/crfm/benchmarking/status.txt`.
1. Exit the screen session: `ctrl+ad`.

Once all the runs complete, visit the
[Benchmarking project status page](https://nlp.stanford.edu/crfm/benchmarking/status.txt).

### To visualize results at crfm-models.stanford.edu

1. Run `venv/bin/benchmark-present --output-path src/proxy/static/benchmark_output`.
1. Visit the [benchmarking status page](https://crfm-models.stanford.edu/static/benchmarking.html).

# Contributing

## One-time setup

To contribute to this project, install the dependencies and git hook scripts:

    ./pre-commit.sh && pre-commit install

## Tests

To run all unit tests:

    python -m pytest

Append `-vv` to output the full diff and results:

    python -m pytest -vv

To run a specific file, simply specify the path:

    python -m pytest <path/to/file> -vv
