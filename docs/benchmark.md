# Running the benchmark

Examples of running the benchmark:

    venv/bin/benchmark-run
    venv/bin/benchmark-run -r mmlu:subject=philosophy
    venv/bin/benchmark-run -r synthetic_reasoning_natural:difficulty=easy
    venv/bin/benchmark-run -r twitter_aae:demographic=aa
    venv/bin/benchmark-run -r copyright:datatag=pilot
    venv/bin/benchmark-run -r disinformation:capability=reiteration
    venv/bin/benchmark-run -r wikifact:k=2,subject=P31
    venv/bin/benchmark-run -r code:dataset=APPS
    venv/bin/benchmark-run -r the_pile:subset=OpenSubtitles
    venv/bin/benchmark-run -r wikifact:subject=P31
    venv/bin/benchmark-run -r raft:subset=ade_corpus_v2
    venv/bin/benchmark-run -r natural_qa:mode=closedbook
    venv/bin/benchmark-run -r natural_qa:mode=openbook-longans
    venv/bin/benchmark-run -r quac
    venv/bin/benchmark-run -r wikitext_103
    venv/bin/benchmark-run -r blimp:phenomenon=irregular_forms
    venv/bin/benchmark-run -r narrative_qa
    venv/bin/benchmark-run -r news_qa
    venv/bin/benchmark-run -r imdb
    venv/bin/benchmark-run -r twitter_aae:demographic=aa

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
1. Run `bash scripts/run-all-stanford.sh --suite v1 --max-eval-instances 1000 --num-threads 8 --priority 2 --local`.
1. After the run for all the models have finished, generate JSON files for the frontend and tables for the paper:
   `benchmark-summarize --suite <Name of the run suite>`.

## Offline evaluation for `TogetherClient` models

### Exporting requests

1. `ssh sc`.
1. Create a screen session: `screen -S together`.
1. Use a john to run the suite: `nlprun --priority high -c 8 -g 0 --memory 64g`.
1. `cd /u/scr/nlp/crfm/benchmarking/benchmarking`.
1. Activate the Conda environment: `conda activate crfm_benchmarking`.
1. Do a dry run to generate `RequestState`s for all the Together models: 
   `bash scripts/generate-together-requests.sh --max-eval-instances 1000 --priority 2 --local`.
1. Exit the screen session: `ctrl+ad`.
1. Check on the dry run by streaming the logs: `tail -f dryrun_<Namne of together model>.log`.
1. The dry run results will be outputted to `benchmark_output/runs/together`.
1. Once the dry run is done, run
   `python3 scripts/together/together_export_requests.py benchmark_output/runs/together prod_env/cache/together.sqlite --output-path requests.jsonl`.
   This command will generate a `requests.jsonl` that contains requests that are not in the cache (`prod_env/cache/together.sqlite`).
1. Upload `requests.jsonl` to CodaLab:
    1. Log on to CodaLab: `cl work main::0xbd9f3df457854889bda8ac114efa8061`.
    1. Upload by running `cl upload requests.jsonl`.
1. Share the link to the CodaLab bundle with our collaborators.

### Importing results

1. `ssh scdt`
1. `cd /u/scr/nlp/crfm/benchmarking/benchmarking`
1. Download the results from CodaLab: `cl download <UUID of the results bundle>`.
1. Run: `python3 scripts/together/together_import_results.py <Path to results jsonl file> prod_env/cache/together.sqlite`.
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
