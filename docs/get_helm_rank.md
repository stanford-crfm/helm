# Get Your Model's Leaderboard Rank

This tutorial will show you how to locally insert your model into the HELM (Core-scenarios) ranking, with in 3 steps:

## Download HELM leaderboard results

First, in order to compare your model to the latest and greatest models found in the [HELM leaderboard](https://crfm.stanford.edu/helm/latest/?group=core_scenarios), use the following command to obtain a zip file of all previous HELM results

```bash
wget file.zip
```

Once downloaded, expand the file into HELM's results directory (`v1` in this case):

```bash
unzip file.zip -d
path_to_your_helm_repo/benchmark_output/runs/v1 
```

## Run Efficient-HELM (Core-scenarios)

According to [Efficient Benchmarking (of Language Models)](https://arxiv.org/pdf/2308.11696.pdf), which systematically analysed benchmark design choices using the HELM benchmark as an example, one can run the HELM benchmark with a fraction of the examples and still get a reliable estimation of a full run (Perlitz et al., 2023).  

Specifically, the authors calculated the CI $95\%$ of Rank Location from the real ranks as a function of the number of examples used per scenario and came up with the following tradeoffs[^1]:

| Examples Per Scenario | CI $95\%$ of Rank Location | Compute saved |
| :-------------------: | :------------------------: | :-----------: |
|         $10$          |           $\pm5$           |  $\times400$  |
|         $20$          |           $\pm4$           |  $\times200$  |
|         $50$          |           $\pm3$           |  $\times80$   |
|         $200$         |           $\pm2$           |  $\times20$   |
|        $1000$         |           $\pm1$           |   $\times4$   |
|          All          |           $\pm1$           |   $\times1$   |

Choose your tradeoff; how accurate do you need your rank? how much time (compute) do you want to wait (spend)? 
Once you have chosen, (for example, if you are OK with $\pm5$) you can obtain results using the following command:

```bash
helm-run --conf-paths run_specs_core_scenarios_10 --suite v1
```

In case you require something more accurate, replace `run_specs_core_scenarios_10` with, for example, `run_specs_core_scenarios_100` etc.

## Summarize and serve your results

To view how your model fits in with the latest leaderboard, process and aggregate your results with:

```bash
helm-summarize --suite v1
```

And serve with:

```bash
helm-server
```

## References List:

```Perlitz, Y., Bandel, E., Gera, A., Arviv, O., Ein-Dor, L., Shnarch, E., Slonim, N., Shmueli-Scheuer, M. and Choshen, L., 2023. Efficient Benchmarking (of Language Models). arXiv preprint arXiv:2308.11696.```

[^1]: Note that the quantities below are the CI $95\%$ of the rank location and are thus very conservative estimates. In our experiments, we did not experience deviations above $\pm2$ for any of the options above.
