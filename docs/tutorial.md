# Tutorial

This tutorial will explain how to use the HELM command line tools to run benchmarks, aggregate statistics, and visualize results.

We will run two runs using the `mmlu` scenario on the `openai/gpt2` model. The `mmlu` scenario implements the **Massive Multitask Language (MMLU)** benchmark from [this paper](https://arxiv.org/pdf/2009.03300.pdf), and consists of a Question Answering (QA) task using a dataset with questions from 57 subjects such as elementary mathematics, US history, computer science, law, and more. Note that GPT-2 performs poorly on MMLU, so this is just a proof of concept. We will run two runs: the first using questions about anatomy, and the second using questions about philosophy.

## Using `helm-run`

`helm-run` is a command line tool for running benchmarks.

To run this benchmark using the HELM command-line tools, we need to specify **run entries** that describes the desired runs. For this example, the run entries are `mmlu:subject=anatomy,model=openai/gpt2` (for anatomy) and `mmlu:subject=philosophy,model=openai/gpt2` (for philosophy).

We will now use `helm-run` to execute the runs. Run this command:

```sh
helm-run --run-entries mmlu:subject=anatomy,model=openai/gpt2 mmlu:subject=philosophy,model=openai/gpt2 --suite my-suite --max-eval-instances 10
```

The meaning of the arguments are as follows:

- `--run-entries` specifies the run entries from the desired runs.
- `--suite` specifies a subdirectory under the output directory in which all the output will be placed.
- `--max-eval-instances` limits evaluation to only *N* instances (i.e. items) from the benchmark, using a randomly shuffled order of instances.

`helm-run` creates an environment directory environment and an output directory by default.

-  The environment directory is `prod_env/` by default and can be set using `--local-path`. Credentials for making API calls should be added to a `credentials.conf` file in this directory.
-  The output directory is `benchmark_output/` by default and can be set using `--output-path`.

After running this command, navigate to the `benchmark_output/runs/my-suite/` directory. This should contain a two sub-directories named `mmlu:subject=anatomy,model=openai_gpt2` and `mmlu:subject=philosophy,model=openai_gpt2`. Note that the names of these sub-directories is based on the run entries we used earlier, but with `/` replaced with `_`.

Each output sub-directory will contain several JSON files that were generated during the corresponding run:

- `run_spec.json` contains the `RunSpec`, which specifies the scenario, adapter and metrics for the run.
- `scenario.json` contains a serialized `Scenario`, which contains the scenario for the run and specifies the instances (i.e. inputs) used.
- `scenario_state.json` contains a serialized `ScenarioState`, which contains every request to and response from the model.
- `per_instance_stats.json` contains a serialized list of `PerInstanceStats`, which contains the statistics produced for the metrics for each instance (i.e. input).
- `stats.json` contains a serialized list of `PerInstanceStats`, which contains the statistics produced for the metrics, aggregated across all instances (i.e. inputs).

## Using `helm-summarize`

The `helm-summarize` reads the output files of `helm-run` and computes aggregate statistics across runs. Run the following:

```sh
helm-summarize --suite my-suite
```

This reads the pre-existing files in `benchmark_output/runs/my-suite/` that were written by `helm-run` previously, and writes the following new files back to `benchmark_output/runs/my-suite/`:

- `summary.json` contains a serialized `ExecutiveSummary` with a date and suite name.
- `run_specs.json` contains the run entries for all the runs.
- `runs.json` contains serialized list of `Run`, which contains the run path, run spec and adapter spec and statistics for each run.
- `groups.json` contains a serialized list of `Table`, each containing information about groups in a group category.
- `groups_metadata.json` contains a list of all the groups along with a human-readable description and a taxonomy.

Additionally, for each group and group-relavent metric, it will output a pair of files: `benchmark_output/runs/my-suite/groups/latex/<group_name>_<metric_name>.tex` and `benchmark_output/runs/my-suite/groups/json/<group_name>_<metric_name>.json`. These files contain the statistics for that metric from each run within the group.

## Using `helm-server`

Finally, the `helm-server` command launches a web server to visualize the output files of `helm-run` and `helm-benchmark`. Run:

```sh
helm-server --suite my-suite
```

Open a browser and go to http://localhost:8000/ to view the visualization. You should see a similar view as [live website for the paper](https://crfm.stanford.edu/helm/classic/latest/), but for the data from your benchmark runs. The website has the following sections accessible from the top menu bar:

- **Leaderboards** contains the leaderboards with aggregate metrics.
- **Models** contains a list of models and their descriptions
- **Scenarios** contains a list of scenarios and their descriptions.
- **Predictions** contains a searchable list of runs.
