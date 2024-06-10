# Tutorial

This tutorial will explain how to use the HELM command line tools to run benchmarks, aggregate statistics, and visualize results.

We will run two runs using the `mmlu` scenario on the `openai/gpt2` model. The `mmlu` scenario implements the **Massive Multitask Language (MMLU)** benchmark from [this paper](https://arxiv.org/pdf/2009.03300.pdf), and consists of a Question Answering (QA) task using a dataset with questions from 57 subjects such as elementary mathematics, US history, computer science, law, and more. Note that GPT-2 performs poorly on MMLU, so this is just a proof of concept. We will run two runs: the first using questions about anatomy, and the second using questions about philosophy.

## Using `helm-run`

`helm-run` is a command line tool for running benchmarks.

To run this benchmark using the HELM command-line tools, we need to specify **run spec descriptions** that describes the desired runs. For this example, the run spec descriptions are `mmlu:subject=anatomy,model=openai/gpt2` (for anatomy) and `mmlu:subject=philosophy,model=openai/gpt2` (for philosophy).

Next, we need to create a **run spec configuration file** containing these run spec descriptions. A run spec configuration file is a text file containing `RunEntries` serialized to JSON, where each entry in `RunEntries` contains a run spec description. The `description` field of each entry should be a **run spec description**. Create a text file named `run_entries.conf` with the following contents:

```
entries: [
  {description: "mmlu:subject=anatomy,model=openai/gpt2", priority: 1},
  {description: "mmlu:subject=philosophy,model=openai/gpt2", priority: 1},
]
```

We will now use `helm-run` to execute the runs that have been specified in this run spec configuration file. Run this command:

```
helm-run --conf-paths run_entries.conf --suite v1 --max-eval-instances 10
```

The meaning of the additional arguments are as follows:

- `--suite` specifies a subdirectory under the output directory in which all the output will be placed.
- `--max-eval-instances` limits evaluation to only the first *N* inputs (i.e. instances) from the benchmark.

`helm-run` creates an environment directory environment and an output directory by default.

-  The environment directory is `prod_env/` by default and can be set using `--local-path`. Credentials for making API calls should be added to a `credentials.conf` file in this directory.
-  The output directory is `benchmark_output/` by default and can be set using `--output-path`.

After running this command, navigate to the `benchmark_output/runs/v1/` directory. This should contain a two sub-directories named `mmlu:subject=anatomy,model=openai_gpt2` and `mmlu:subject=philosophy,model=openai_gpt2`. Note that the names of these sub-directories is based on the run spec descriptions we used earlier, but with `/` replaced with `_`.

Each output sub-directory will contain several JSON files that were generated during the corresponding run:

- `run_spec.json` contains the `RunSpec`, which specifies the scenario, adapter and metrics for the run.
- `scenario.json` contains a serialized `Scenario`, which contains the scenario for the run and specifies the instances (i.e. inputs) used.
- `scenario_state.json` contains a serialized `ScenarioState`, which contains every request to and response from the model.
- `per_instance_stats.json` contains a serialized list of `PerInstanceStats`, which contains the statistics produced for the metrics for each instance (i.e. input).
- `stats.json` contains a serialized list of `PerInstanceStats`, which contains the statistics produced for the metrics, aggregated across all instances (i.e. inputs).

`helm-run` provides additional arguments that can be used to filter out `--models-to-run`, `--groups-to-run` and `--priority`. It can be convenient to create a large `run_entries.conf` file containing every run spec description of interest, and then use these flags to filter down the RunSpecs to actually run. As an example, the main `run_specs.conf` file used for the HELM benchmarking paper can be found [here](https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/presentation/run_specs.conf).

**Using model or model_deployment:** Some models have several deployments (for exmaple `eleutherai/gpt-j-6b` is deployed under `huggingface/gpt-j-6b`, `gooseai/gpt-j-6b` and `together/gpt-j-6b`). Since the results can differ depending on the deployment, we provide a way to specify the deployment instead of the model. Instead of using `model=eleutherai/gpt-g-6b`, use `model_deployment=huggingface/gpt-j-6b`. If you do not, a deployment will be arbitrarily chosen. This can still be used for models that have a single deployment and is a good practice to follow to avoid any ambiguity.

## Using `helm-summarize`

The `helm-summarize` reads the output files of `helm-run` and computes aggregate statistics across runs. Run the following:

```
helm-summarize --suite v1
```

This reads the pre-existing files in `benchmark_output/runs/v1/` that were written by `helm-run` previously, and writes the following new files back to `benchmark_output/runs/v1/`:

- `summary.json` contains a serialized `ExecutiveSummary` with a date and suite name.
- `run_specs.json` contains the run spec descriptions for all the runs.
- `runs.json` contains serialized list of `Run`, which contains the run path, run spec and adapter spec and statistics for each run.
- `groups.json` contains a serialized list of `Table`, each containing information about groups in a group category.
- `groups_metadata.json` contains a list of all the groups along with a human-readable description and a taxonomy.

Additionally, for each group and group-relavent metric, it will output a pair of files: `benchmark_output/runs/v1/groups/latex/<group_name>_<metric_name>.tex` and `benchmark_output/runs/v1/groups/json/<group_name>_<metric_name>.json`. These files contain the statistics for that metric from each run within the group.

<!--
# TODO(#1441): Enable plots

## Using `helm-create-plots`

The `helm-create-plots` reads the `groups` directory created by `helm-summarize` and creates plots, equivalent to those use in the HELM paper. Run the following:

```
helm-create-plots --suite v1
```

This reads the pre-existing files in `benchmark_output/runs/v1/groups` that were written by `helm-summarize` previously,
and creates plots (`.png` or `.pdf`) at `benchmark_output/runs/v1/plots`.

-->

## Using `helm-server`

Finally, the `helm-server` command launches a web server to visualize the output files of `helm-run` and `helm-benchmark`. Run:

```
helm-server
```

Open a browser and go to http://localhost:8000/ to view the visualization. You should see a similar view as [live website for the paper](https://crfm.stanford.edu/helm/v1.0/), but for the data from your benchmark runs. The website has three main sections:

- **Models** contains a list of available models.
- **Scenarios** contains a list of available scenarios.
- **Results** contains results from the runs, organized into groups and categories of groups.
- **Raw Runs** contains a searchable list of runs.

## Other Tips

- The suite name can be used as a versioning mechanism to separate runs using different versions of scenarios or models.
- Tools such as [`jq`](https://stedolan.github.io/jq/) are useful for examining the JSON output files on the command line.
