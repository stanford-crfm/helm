# Get Your Model's Leaderboard Rank

This tutorial will show you how to locally add your model into the HELM leaderboard, with in 3 steps:

## Download HELM leaderboard results

First, in order to compare your model to the latest and greatest models found in the [HELM leaderboard](https://crfm.stanford.edu/helm/latest/?group=core_scenarios), use the following command to obtain a zip file of all previous HELM results

```bash
export LEADERBOARD_VERSION=v0.3.0
```

Downloaded, expand the file into HELMs results dir:

```bash
curl -O https://storage.googleapis.com/crfm-helm-public/benchmark_output/archives/$LEADERBOARD_VERSION/run_stats.zip &&\
mkdir -p benchmark_output/runs/$LEADERBOARD_VERSION && unzip run_stats.zip -d benchmark_output/runs/$LEADERBOARD_VERSION
```

now that the files are in your results directory, all HELM models will be shown in your UI along with your model.

## Run Efficient-HELM

According to [Efficient Benchmarking (of Language Models)](https://arxiv.org/pdf/2308.11696.pdf) a paper from IBM Research, which systematically analysed benchmark design choices using the HELM benchmark as an example, one can run the HELM benchmark with a fraction of the examples and still get a reliable estimation of a full run (Perlitz et al., 2023).  

Specifically, the authors calculated the CI 95% of Rank Location from the real ranks as a function of the number of examples used per scenario and came up with the following tradeoffs[^1]:

| Examples Per Scenario | CI 95% of Rank Location | Compute saved |
| :-------------------: | :---------------------: | :-----------: |
|          10           |           ±5            |     X400      |
|          20           |           ±4            |     X200      |
|          50           |           ±3            |      X80      |
|          200          |           ±2            |      X20      |
|         1000          |           ±1            |      X4       |
|          All          |           ±1            |      X1       |


Choose your point on your tradeoff, how accurate do you need your rank? how much time do you want to wait? Once you have chosen, download the config and define your model
```bash
export EXAMPLES_PER_SCENARIO=10 && \
export MODEL_TO_RUN=openai/gpt2
```

That's it, run the following to get the config file:

```bash
wget https://raw.githubusercontent.com/stanford-crfm/helm/main/src/helm/benchmark/presentation/run_entries_core_scenarios_$EXAMPLES_PER_SCENARIO.conf -O run_entries_$EXAMPLES_PER_SCENARIO.conf
```

and this one to run the benchmark (will take some time in the first time since all the data has to be prepared):

```bash
helm-run \
--conf-paths run_entries_$EXAMPLES_PER_SCENARIO.conf \
--suite $LEADERBOARD_VERSION \
--max-eval-instances $EXAMPLES_PER_SCENARIO \
--models-to-run $MODEL_TO_RUN \
--cache-instances \
--num-train-trials 1 \
--skip-completed-runs
```

This will take some time the first time running since all the data (regardless of the number of examples chosen) is downloaded and prepared.


## Summarize and serve your results

To view how your model fits in with the latest leaderboard, process and aggregate your results with:

```bash
helm-summarize --suite $LEADERBOARD_VERSION
```

And serve with:

```bash
helm-server
```

## References List:

```Perlitz, Y., Bandel, E., Gera, A., Arviv, O., Ein-Dor, L., Shnarch, E., Slonim, N., Shmueli-Scheuer, M. and Choshen, L., 2023. Efficient Benchmarking (of Language Models). arXiv preprint arXiv:2308.11696.```

[^1]: Note that the quantities below are the CI 95% of the rank location and are thus very conservative estimates. In our experiments, we did not experience deviations above ±2 for any of the options above.]:
