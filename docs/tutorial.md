# Tutorial

## Setup

### Create a virtual environment

It is recommended to install HELM into a virtual environment with Python version 3.8 to avoid dependency conflicts. HELM requires Python version 3.8. To create, a Python virtual environment with Python version >= 3.8 and activate it, follow the instructions below.

Using [Virtualenv](https://docs.python.org/3/library/venv.html#creating-virtual-environments):

```
# Create a virtual environment.
# Only run this the first time.
python3 -m pip install virtualenv
python3 -m virtualenv -p python3 crfm_helm

# Activate the virtual environment.
source crfm_helm/bin/activate
```

Using [Anaconda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html):

```
# Create a virtual environment.
# Only run this the first time.
conda create -n crfm-helm python=3.8 pip

# Activate the virtual environment.
conda activate crfm-helm
```

### Install HELM

Within this virtual environment, run:

```
pip install crfm-helm
```

## Quick Start

Run the following:

```
# Create a run specs configuration
echo 'entries: [{description: "mmlu:subject=philosophy,model=huggingface/gpt2", priority: 1}]' > run_specs.conf

# Run benchmark
helm-run --conf-paths run_specs.conf --local --suite v1 --max-eval-instances 10

# Summarize benchmark results
helm-summarize --suite v1

# Start a web server to display benchmark results
helm-server
```

Then go to http://localhost:8000/ in your browser.

## Command-line Tools Tutorial

This tutorial will explain how to use the HELM command line tools to run benchmarks, aggregate statistics, and visualize results.

In this tutorial, we will run two runs using the `mmlu` scenario on the `huggingface/gpt-2` model. The `mmlu` scenario implements the **Massive Multitask Language (MMLU)** benchmark from [this paper](https://arxiv.org/pdf/2009.03300.pdf), and consists of a Question Answering (QA) task using a dataset with questions from 57 subjects such as elementary mathematics, US history, computer science, law, and more. We will run two runs: the first using questions about anatomy, and the second using questions about philosophy.

### Using `helm-run`

`helm-run` is a command line tool for running benchmarks.

To run this benchmark using the HELM command-line tools, we need to specify **run spec descriptions** that describes the desired runs. For this example, the run spec descriptions are `mmlu:subject=anatomy,model=huggingface/gpt-2` (for anatomy) and `mmlu:subject=philosophy,model=huggingface/gpt-2` (for philosophy).

Next, we need to create a **run spec configuration file** contining these run spec descriptions. A run spec configuration file is a text file containing `RunEntries` serialized to JSON, where each entry in `RunEntries` contains a run spec description. The `description` field of each entry should be a **run spec description**. Create a text file named `run_spec.conf` with the following contents:

```
entries: [
  {description: "mmlu:subject=anatomy,model=huggingface_gpt-2", priority: 1},
  {description: "mmlu:subject=philosophy,model=huggingface_gpt-2", priority: 1},
]
```

We will now use `helm-run` to execute the runs that have been specified in this run spec configuration file. Run this command:

```
helm-run --run-specs run_spec.conf --local --suite v1 --max-eval-instances 10
```

The meaning of the additional arguments are as follows:

- `--local` specifies not to use the Stanford CRFM API. You should always use `--local` unless you are a Stanford CRFM API user.
- `--suite` specifies a subdirectory under the output directory in which all the output will be placed.
- `--max-eval-instances` limits evaluation to only the first *N* inputs (i.e. instances) from the benchmark.

`helm-run` creates an environment directory environment and an output directory by default.

-  The environment directory is `prod_env/` by default and can be set using `--local-path`.
-  The output directory is `benchmarking_output/` by default and can be set using `--output-path`.

After running this command, navigate to the `benchmarking_output/runs/v1/` directory. This should contain a two sub-directories named `mmlu:subject=anatomy,model=huggingface_gpt-2` and `mmlu:subject=philosophy,model=huggingface_gpt-2`. Note that the names of these sub-directories is based on the run spec descriptions we used earlier, but with `/` replaced with `_`.

Each output sub-directory will contain several JSON files that were generated during the corresponding run:

- `per_instance_stats.json` contains a serialized list of `PerInstanceStats`, which contains the statistics produced for the metrics for each instance (i.e. input).
- `run_spec.json` contains the `RunSpec`, which specifies the scenario, adapter and metrics for the run.
- `scenario.json` contains a serialized `Scenario`, which contains the scenario for the run and specifies the instances (i.e. inputs) used.
- `scenario_state.json` contains a serialized `ScenarioState`, which contains every request to and response from the model.
- `stats.json` contains a serialized list of `PerInstanceStats`, which contains the statistics produced for the metrics, aggregated across all instances (i.e. inputs).

The command also writes the file `benchmarking_output/runs/v1/run_specs.json`, which contains the run spec descriptions for all the runs. 

`helm-run` provides additional arguments that can be used to filter out  `--models-to-run`, `--groups-to-run` and `--priority`. It can be convenient to create a large `run_specs.conf` file containing every run spec description of interest, and then use these flags to filter down the RunSpecs to actually run. As an example, the main `run_specs.conf` file used for the HELM benchmarking paper can be found [here](https://github.com/stanford-crfm/helm/blob/main/src/benchmark/presentation/run_specs.conf).

### Using `helm-summarize`

The `helm-summarize` reads the output files of `helm-run` and computes aggregate statistics. Run the following:

```
helm-summarize --suite v1
```

This reads the pre-existing files in `benchmark_output/runs/v1/` that were written by `helm-run` previously, and writes the following new files back to `benchmark_output/runs/v1/`:

- `summary.json` contains a serialized `ExecutiveSummary` with a date and suite name.
- `runs.json` contains serialized list of `Run`, which contains the run path, run spec and adapter spec and statistics for each run.
- `groups.json` contains a serialized list of `Table`, each containing information about groups in a group category.
- `groups_metadata.json` contains a list of all the groups along with a human-readable description and a taxonomy.

Additionally, for each group and group-relavent metric, it will output a pair of files: `benchmark_output/runs/v1/groups/latex/<group_name>_<metric_name>.tex` and `benchmark_output/runs/v1/groups/latex/<group_name>_<metric_name>.json`. These files contain the statistics for that metric from each run within the group.

**Note**: Because `helm-summarize` reads the `run_specs.json` that was produced by `helm-run` and `run_specs.json` is overwritten by every invocation of `helm-run`, `helm-summarize` will only process runs from the most recent invocation of `helm-run`.

### Using `helm-server`

Finally, the `helm-server` command launches a web server to visualize the output files of `helm-run` and `helm-benchmark`. Run:

```
helm-server
```

Open a browser and go to http://localhost:8000/ to view the visualization. You should see a similar view as [live website for the paper](https://crfm.stanford.edu/helm/v1.0/), but for the data from your benchmark runs. The website has three main sections:

- **Models** contains a list of available models.
- **Groups** contains results from the runs, organized into groups and categories of groups.
- **Runs** contains a searchable list of runs.

### Other Tips

- The suite name can be used as a versioning mechanism to separate runs using different versions of scenarios or models.
- Tools such as [`jq`](https://stedolan.github.io/jq/) are useful for examining the JSON output files on the command line.

## Constructing RunSpec descriptions

### RunSpecs Descriptions

A **RunSpec description** is a string that describes one or more **RunSpecs**. Each **RunSpec** specifies how to run a single **run**. For most uses of HELM command-line tools, the RunSpec should be specified in the form of a RunSpec description.

A RunSpec description contains **scenario name** for the scenario to run, and one or more **parameter key-value pairs**. The syntax of a RunSpec description is:`<scenario_name>:<parameter_key>=<parameter_value>,<parameter_key>=<parameter_value>`.

An example RunSpec description is `mmlu:subject=philosophy,model=openai/text-davinci-002`. This RunSpec description describes a single RunSpec with the `mmlu` scenario with the parameter `subject=philosophy` that is passed through a `model` run expander with the parameter `openai/text-davinci-002`.

### Scenarios

A **scenario** is a **task** and a **dataset** that can be run on a model. For instance, the `mmlu` scenario consists of a Question Answering (QA) task using the the Massive Multitask Language (MMLU) dataset.

Each scenario is also associated with a default **adapter** and default **metrics**. An adapter converts **input instances** into the actual requests that will be sent to the model. The metrics take the responses from the model and compute **statistics** over the responses.

Some scenarios have required and optional parameters that should be specified in the RunSpec description. For instance, the `mmlu` scenario has a required `subject` parameter and an optional `method` parameter. These parameters should be specified as described in the RunSpec description above e.g. `subject=philosophy`. If a required parameter is omitted, an error will result.

Refer to the documentation in the corresponding `Scenario` class for each scenario's arguments, as well for descriptions of scenarios and references to the original papers and datasets.

### Run Expanders

A **`RunExpander`**, takes a RunSpec as input and outputs one or more generated RunSpecs.

After the scenario name and scenario parameters are used to construct a **base RunSpec**, the remaining parameters in the RunSpec description are used to run **`RunExpanders`** that produce **generated RunSpecs**. The generated RunSpecs (_not_ the base RunSpec) determine the runs that get executed by the HELM command-line tools.

The parameter key specifies which `RunExpander` to run. For instance, the `model` key corresponds to `ModelRunExpander`. The parameter value is passed into the `RunExpander` as an argument.

Some `RunExpanders` can result in multiple RunSpecs. For instance, if the RunSpec description contained `model=text`, `ModelRunExpander` would generate a RunSpec for each available text model.

Refer to the documentation in the corresponding `RunExpander` class for the arguments and behavior of each run expander.

Generally, every RunSpec description should specify at least the `model` parameter. This determines which model(s) will be used for the run. The value of the parameter should be the name of a specific model, or the name of a category of models.
