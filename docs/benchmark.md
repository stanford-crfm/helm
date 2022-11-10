# Running the benchmark

In the following, assume that the suite (the directory where everything is written) is:

    export SUITE=v1

Some of the benchmarks (NewsQA) depend on data that's not public: all such data
will be stored in the `restricted` directory.  You need to make sure that
directory exists.

To try to test things out a small subset (defined in `run_specs_small.conf`) with just 10 eval instances:

    # Just load the config file
    venv/bin/benchmark-present --conf src/benchmark/presentation/run_specs_small.conf --local --max-eval-instances 10 --suite $SUITE --skip-instances

    # Create the instances and the requests, but don't execute
    venv/bin/benchmark-present --conf src/benchmark/presentation/run_specs_small.conf --local --max-eval-instances 10  --suite $SUITE --dry-run

    # Execute the requests and compute metrics
    venv/bin/benchmark-present --conf src/benchmark/presentation/run_specs_small.conf --local --max-eval-instances 10  --suite $SUITE

    # Generate assets for the website
    venv/bin/benchmark-summarize --suite $SUITE

Notes:
- `--local` means we bypass the proxy server.
- All the outputs should be in `benchmark_output/runs/$SUITE`.

To run everything (note we're restricting the number of instances and
scenarios) in parallel:

    # Generate all the commands to run in parallel
    venv/bin/benchmark-present --local --suite $SUITE --max-eval-instances 1000 --priority 2 --num-threads 8 --skip-instances

    # Run everything in parallel over Slurm
    bash benchmark_output/runs/$SUITE/run-all.sh

    # Wait for all Slurm jobs to finish, monitor the logs
    # tail benchmark_output/runs/$SUITE/slurm-*.out

    # Generate assets for the website
    venv/bin/benchmark-present --local --suite $SUITE --max-eval-instances 1000 --skip-instances
    venv/bin/benchmark-summarize --suite $SUITE

    # Run a simple Python server to make sure things work at http://localhost:8000
    benchmark-server

    # Copy all website assets to the `www` directory, which can be copied to GitHub pages for static serving.
    sh scripts/create-www.sh $SUITE

This should push to the [public site](https://nlp.stanford.edu/$USER/benchmarking/).

Once everytihng has been sanity checked, push `www` to a GitHub page.

## To estimate token usage

To estimate token usage without making any requests, append the `--dry-run` option:

    venv/bin/benchmark-run -r <RunSpec to estimate token usage> --suite $SUITE --max-eval-instances <Number of eval instances> --dry-run

and check the output in `benchmark_output/runs/$SUITE`.


where `sum` indicates the estimated total number of tokens used for the specific `RunSpec`.

For the OpenAI models, we use a
[GPT-2 Tokenizer](https://github.com/stanford-crfm/benchmarking/blob/master/src/proxy/tokenizer/openai_token_counter.py#L12)
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
