# Advanced Benchmarking Guide

## Running Restricted Benchmarks

Some of the benchmarks (NewsQA) depend on data that's not public: all such data
will be stored in the `restricted` directory.  You need to make sure that
directory exists.

## Dry Runs

The `helm-run` provides several flags that can be used to test that the configuration and scenario are working correctly without actually sending requests to the model

    # Just load the config file
    helm-run --conf src/helm/benchmark/presentation/run_specs_small.conf --local --max-eval-instances 10 --suite v1 --skip-instances

    # Create the instances and the requests, but don't send requests to the model
    helm-run --conf src/helm/benchmark/presentation/run_specs_small.conf --local --max-eval-instances 10  --suite v1 --dry-run

## Estimating Token Usage

To estimate token usage without making any requests, append the `--dry-run` option:

    helm-run -r <RunSpec to estimate token usage> --suite $SUITE --max-eval-instances <Number of eval instances> --dry-run

and check the output in `benchmark_output/runs/$SUITE`.

`sum` indicates the estimated total number of tokens used for the specific `RunSpec`.

For the OpenAI models, we use a
[GPT-2 Tokenizer](https://github.com/stanford-crfm/benchmarking/blob/master/src/helm/proxy/tokenizer/openai_token_counter.py#L12)
to estimate the token usage. The tokenizer will be downloaded and cached when running a dry run.

## Perspective API

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
