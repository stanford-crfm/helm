# Running the benchmark

Examples of running the benchmark:

    venv/bin/benchmark-run
    venv/bin/benchmark-run -r mmlu:subject=philosophy --suite SUITE_NAME
    venv/bin/benchmark-run -r synthetic_reasoning_natural:difficulty=easy --suite SUITE_NAME
    venv/bin/benchmark-run -r twitter_aae:demographic=aa --suite SUITE_NAME
    venv/bin/benchmark-run -r copyright:datatag=pilot --suite SUITE_NAME
    venv/bin/benchmark-run -r disinformation:capability=reiteration --suite SUITE_NAME
    venv/bin/benchmark-run -r wikifact:k=2,subject=P31 --suite SUITE_NAME
    venv/bin/benchmark-run -r code:dataset=APPS --suite SUITE_NAME
    venv/bin/benchmark-run -r the_pile:subset=OpenSubtitles --suite SUITE_NAME
    venv/bin/benchmark-run -r wikifact:subject=P31 --suite SUITE_NAME
    venv/bin/benchmark-run -r raft:subset=ade_corpus_v2 --suite SUITE_NAME
    venv/bin/benchmark-run -r natural_qa:mode=closedbook --suite SUITE_NAME
    venv/bin/benchmark-run -r natural_qa:mode=openbook-longans --suite SUITE_NAME
    venv/bin/benchmark-run -r quac --suite SUITE_NAME
    venv/bin/benchmark-run -r wikitext_103 --suite SUITE_NAME
    venv/bin/benchmark-run -r blimp:phenomenon=irregular_forms --suite SUITE_NAME
    venv/bin/benchmark-run -r narrative_qa --suite SUITE_NAME
    venv/bin/benchmark-run -r news_qa --suite SUITE_NAME
    venv/bin/benchmark-run -r imdb --suite SUITE_NAME
    venv/bin/benchmark-run -r twitter_aae:demographic=aa --suite SUITE_NAME

You can also run the benchmark using a local proxy, in which case you have to
first start a local server (see instructions above for more details).

## To estimate token usage

To estimate token usage without making any requests, append the `--dry-run` option:

    venv/bin/benchmark-run -r <RunSpec to estimate token usage> --dry-run

For example, running `venv/bin/benchmark-run -r real_toxicity_prompts --dry-run` outputs:

```text
  Stats {
    MetricName(name='estimated_num_tokens_cost', k=None, split=None, sub_split=None, perturbation=None)[min=505.000, mean=514.957, max=536.000, sum=514957.000 (1000)]
  }
```

where `sum` indicates the estimated total number of tokens used for the specific `RunSpec`.

For the OpenAI models, we use a
[GPT-2 Tokenizer](https://github.com/stanford-crfm/benchmarking/blob/master/src/proxy/tokenizer/openai_token_counter.py#L12)
to estimate the token usage. The tokenizer will be downloaded and cached when running a dry run.

## Final benchmarking (Infrastructure team only)

1. `ssh sc`.
1. Go to the source code directory: `cd /u/scr/nlp/crfm/benchmarking/benchmarking`.
   We have 700 GB of disk space total on `/u/scr/nlp/crfm`.
1. Pull the latest changes: `git pull`.
1. Activate the Conda environment: `conda activate crfm_benchmarking`
   1. Run `./pre-commit.sh` if there are new dependencies to install.
1. Run `bash scripts/run-all-stanford.sh --suite <Suite name>` e.g.,
   `bash scripts/run-all-stanford.sh --suite v1`.
1. After the run for all the models has finished, run the remaining commands the script outputs.

## Offline evaluation

### Exporting requests

1. `ssh sc`.
1. Go to the source code directory: `cd /u/scr/nlp/crfm/benchmarking/benchmarking`.
1. Pull the latest changes: `git pull`.
1. Activate the Conda environment: `conda activate crfm_benchmarking`
   1. Run `./pre-commit.sh` if there are new dependencies to install.
1. Run `bash scripts/run-all-stanford.sh --suite <Suite name> --dry-run` e.g.,
   `bash scripts/run-all-stanford.sh --suite v4-dryrun --dry-run`.
1. Once the dry run is done, run the following commands:
    1. `python3 scripts/offline_eval/export_requests.py together benchmark_output/runs/v4-dryrun 
       --output-path benchmark_output/runs/v4-dryrun/together_requests.jsonl`
    1. `python3 scripts/offline_eval/export_requests.py microsoft benchmark_output/runs/v4-dryrun 
       --output-path benchmark_output/runs/v4-dryrun/microsoft_requests.jsonl`
1. Upload requests JSONL files to CodaLab:
    1. Log on to CodaLab: `cl work main::0xbd9f3df457854889bda8ac114efa8061`.
    1. Upload by Together requests: `cl upload benchmark_output/runs/v4-dryrun/together_requests.jsonl`.
    1. Upload by MT-NLG requests: `cl upload benchmark_output/runs/v4-dryrun/ microsoft_requests.jsonl`.
1. Share the link to the CodaLab bundles with our collaborators.

### Importing results

1. `ssh scdt`
1. `cd /u/scr/nlp/crfm/benchmarking/benchmarking`
1. Download the results from CodaLab: `cl download <UUID of the results bundle>`.
1. Run: `python3 scripts/offline_eval/import_results.py <Org> <Path to results jsonl file>` e.g.,
   `python3 scripts/offline_eval/import_results.py together results.jsonl`.
   This will update the cache with requests and their results.

## To visualize results at crfm-models.stanford.edu

1. Run `venv/bin/benchmark-present --output-path src/proxy/static/benchmark_output`.
1. Visit the [benchmarking status page](https://crfm-models.stanford.edu/static/benchmarking.html).

### To verify that the Scenario construction and generation of prompts are reproducible

1. `ssh scdt`.
1. `cd /u/scr/nlp/crfm/benchmarking/benchmarking`.
1. Create a screen session: `screen -S reproducible`.
1. `conda activate crfm_benchmarking`.
1. Run `python3 scripts/verify_reproducibility.py --models-to-run openai/davinci openai/code-cushman-001 together/gpt-neox-20b
   --conf-path src/benchmark/presentation/run_specs.conf --max-eval-instances 1000 --priority 2 &> reproducible.log`.
1. Check the result at `reproducible.log`.
