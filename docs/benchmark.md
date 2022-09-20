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

Go to the [local website](http://localhost:1959/static/benchmarking.html) to look at the results,
assuming you have run `proxy-server` first.

    # Copy all website assets to a `www` directory for static serving.
    sh scripts/create-www.sh $SUITE

For the final version, push `www` to a GitHub repo.

For debugging, go to the [public site](https://nlp.stanford.edu/pliang/benchmarking/).

## To estimate token usage

To estimate token usage without making any requests, append the `--dry-run` option:

    venv/bin/benchmark-run -r <RunSpec to estimate token usage> --suite $SUITE --max-eval-instances <Number of eval instances> --dry-run

and check the output in `benchmark_output/runs/$SUITE`.


where `sum` indicates the estimated total number of tokens used for the specific `RunSpec`.

For the OpenAI models, we use a
[GPT-2 Tokenizer](https://github.com/stanford-crfm/benchmarking/blob/master/src/proxy/tokenizer/openai_token_counter.py#L12)
to estimate the token usage. The tokenizer will be downloaded and cached when running a dry run.

## Final benchmarking (Infrastructure team only)

1. `ssh sc`.
1. Go to the source code directory: `cd /nlp/scr2/nlp/crfm/benchmarking/benchmarking`.
   We have 2 TB of disk space total on `/nlp/scr2/nlp/crfm`.
1. Pull the latest changes: `git pull`.
1. Activate the Conda environment: `conda activate crfm_benchmarking`
   1. Run `./pre-commit.sh` if there are new dependencies to install.
1. Run `bash scripts/run-all-stanford.sh --suite <Suite name>` e.g.,
   `bash scripts/run-all-stanford.sh --suite v6`.
1. After the run for all the models has finished, run the remaining commands the script outputs.

## Offline evaluation

### Exporting requests

1. `ssh sc`.
1. Go to the source code directory: `cd /nlp/scr2/nlp/crfm/benchmarking/benchmarking`.
1. Pull the latest changes: `git pull`.
1. Activate the Conda environment: `conda activate crfm_benchmarking`
   1. Run `./pre-commit.sh` if there are new dependencies to install.
1. Run `bash scripts/run-all-stanford.sh --suite <Suite name> --dry-run` e.g.,
   `bash scripts/run-all-stanford.sh --suite v6-dryrun --dry-run`.
1. Once the dry run is done, run the following commands:
    1. `python3 scripts/offline_eval/export_requests.py together benchmark_output/runs/v6-dryrun
       --output-path benchmark_output/runs/v6-dryrun/together_requests.jsonl`
    1. `python3 scripts/offline_eval/export_requests.py microsoft benchmark_output/runs/v6-dryrun
       --output-path benchmark_output/runs/v6-dryrun/microsoft_requests.jsonl`
1. Upload requests JSONL files to CodaLab:
    1. Log on to CodaLab: `cl work main::0xbd9f3df457854889bda8ac114efa8061`.
    1. Upload by Together requests: `cl upload benchmark_output/runs/v6-dryrun/together_requests.jsonl`.
    1. Upload by MT-NLG requests: `cl upload benchmark_output/runs/v6-dryrun/microsoft_requests.jsonl`.
1. Share the link to the CodaLab bundles with our collaborators.

### Importing results

1. `ssh scdt`
1. `cd /nlp/scr2/nlp/crfm/benchmarking/benchmarking`
1. Download the results from CodaLab: `cl download <UUID of the results bundle>`.
1. Run: `python3 scripts/offline_eval/import_results.py <Org> <Path to results jsonl file>` e.g.,
   `python3 scripts/offline_eval/import_results.py together results.jsonl`.
   This will update the cache with requests and their results.

### To verify that the Scenario construction and generation of prompts are reproducible

1. `ssh scdt`.
1. `cd /nlp/scr2/nlp/crfm/benchmarking/benchmarking`.
1. Create a screen session: `screen -S reproducible`.
1. `conda activate crfm_benchmarking`.
1. Run `python3 scripts/verify_reproducibility.py --models-to-run openai/davinci openai/code-cushman-001 together/gpt-neox-20b
   --conf-path src/benchmark/presentation/run_specs.conf --max-eval-instances 1000 --priority 2 &> reproducible.log`.
1. Check the result at `reproducible.log`.
