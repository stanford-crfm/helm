# Tutorial

## Installation

It is recommended to install HELM into a virtual environment that is created using a tool such as [Conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or [virtualenv](https://virtualenv.pypa.io/).

Within this virtual environment, run:

```
pip install crfm-helm
```

## Command-line Tools

### Using `helm-run`

For this example, we will run the Massive Multitask Language (MMLU) benchmark from [this paper](https://arxiv.org/pdf/2009.03300.pdf) on the `huggingface/gpt-2` model. The MMLU scenario is implemented in `MMLUScenario`, which has a constructor that takes in a single argument `subject`. We will use `philosophy` as our subject.

To run this benchmark using the HELM command-line tools, we need to pass in a **RunSpec description** that describes the desired run(s). For this example, the RunSpec description is `mmlu:subject=philosophy,model=huggingface/gpt-2`.

`helm-run` is a command-line tool for executing a single RunSpec description. Run this command:

```
helm-run --run-specs mmlu:subject=philosophy,model=huggingface/gpt2 --local --suite v1 --max-eval-instances 10
```

The meaning of the additional arguments are as follows:

- `--local` specifies not to use the Stanford CRFM API. You should always use `--local` unless you are a Stanford CRFM API user.
- `--suite` specifies a subdirectory under the output directory in which all the output will be placed.
- `--max-eval-instances` limits evaluation to only the first N inputs (i.e. instances) from the benchmark.

`helm-run` creates an environment directory environment and an output directory by default.

-  The environment directory is `./prod_env/` by default and can be set using `--local-path`.
-  The output directory is `./benchmarking_output/` by default and can be set using `--output-path`.

After running this command, navigate to the `benchmarking_output/runs/v1/` directory. This should contain a subdirectory named `mmlu:subject=philosophy,model=huggingface_gpt-2`. Note that the name of this subdirectory is based on the RunSpec description we used earlier, but with `/` replaced with `_`.

This output directory will contain several JSON files:

- `per_instance_stats.json` contains a serialized list of `PerInstanceStats`, which contains the statistics produced for the metrics for each instance (i.e. input).
- `run_spec.json` contains the `RunSpec`, which specifies the scenario, adapter and metrics for the run.
- `scenario.json` contains a serialized `Scenario`, which contains the scenario for the run and specifies the instances (i.e. inputs) used.
- `scenario_state.json` contains a serialized `ScenarioState`, which contains every request to and response from the model.
- `stats.json` contains a serialized list of `PerInstanceStats`, which contains the statistics produced for the metrics, aggregated across all instances (i.e. inputs).

To examine the JSON files, you can use a tool such as [`jq`](https://stedolan.github.io/jq/), or you can use `helm-summarize` to generate a web report that you can view in your browser, as explained later.

`helm-run` also can also execute multiple RunSpec descriptions. For instance, the following command runs two instances of the MMLU benchmark, one each for the "anatomy" and "philosophy" subjects.

```
helm-run --run-specs mmlu:subject=anatomy,model=huggingface/gpt2 --run-specs mmlu:subject=philosophy,model=huggingface/gpt2  --local --suite v1 --max-eval-instances 10
```
After running this, the `./benchmarking_output/runs/v1/` directory should contain two subdirectories, `mmlu:subject=anatomy,model=huggingface_gpt-2` and `mmlu:subject=philosophy,model=huggingface_gpt-2`. The contents of each subdirectory should be the same as described previously.

### Using `helm-present`

It is often more convenient to have RunSpec descriptions specified in a configuration file. To execute multiple RunSpec descriptions, create a `run_specs.conf` file containing a `RunEntries` serialized to JSON. The `description` field of each entry should be a RunSpec description. Example:

```
entries: [
  {description: "mmlu:subject=anatomy,model=huggingface_gpt-2", priority: 1},
  {description: "mmlu:subject=philosophy,model=huggingface_gpt-2", priority: 1},
]
```

Then run:

```
helm-present --conf src/benchmark/presentation/run_specs_msmarco.conf --local --suite v1 --max-eval-instances 10
```

After running this, the `benchmarking_output/runs/v1/` directory should contain two subdirectories, `mmlu:subject=anatomy,model=huggingface_gpt-2` and `mmlu:subject=philosophy,model=huggingface_gpt-2`. The contents of each subdirectory should be the same as described previously.

The `helm-present` commands supports nearly all of the same arguments as `helm-run`. Additionally, `helm-present`provides additional arguments that filter which `RunSpecs` get run: `--models-to-run`, `--groups-to-run` and `--priority`. It can be convenient to create a large `run_specs.conf` file containing every RunSpec description of interest, and then use these flags to filter down the RunSpecs to actually run. As an example, the main `run_specs.conf` file used for the HELM benchmarking paper can be found [here](https://github.com/stanford-crfm/helm/blob/main/src/benchmark/presentation/run_specs.conf).

## Constructing RunSpec descriptions

### RunSpecs Descriptions

A **RunSpec description** is a string that describes one or more **RunSpecs**. Each **RunSpec** specifies how to run a single **run**. For most uses of HELM command-line tools, the RunSpec should be specified in the form of a RunSpec description.

A RunSpec description contains **scenario name** for the scenario to run, and one or more **parameter key-value pairs**. The syntax of a RunSpec description is:`<scenario_name>:<parameter_key>=<parameter_value>,<parameter_key>=<parameter_value>`.

An example RunSpec description is `mmlu:subject=philosophy,model=openai/text-davinci-002`. This RunSpec description describes a single RunSpec with the `mmlu` scenario with the parameter `subject=philosophy` that is passed through a `model` run expander with the parameter `openai/text-davinci-002`.

### Scenarios

A **scenario** is a **task** and a **dataset** that can be run on a model. For instance, the `mmlu` scenario consists of a Question Answering (QA) task using the the Massive Multitask Language (MMLU) dataset.

Each scenario is also associated with a default **adapter** and default **metrics**. An adapter converts **input instances** into the actual requests that will be sent to the model. The metrics take the responses from the model and compute **statistics** over the responses.

Some scenarios have required and optional parameters that should be specified in the RunSpec description. For instance, the `mmlu` scenario has a required `subject` parameter and an optional `method` parameter. These parameters should be specified as described in the RunSpec description above e.g. `subject=philosophy`. If a required parameter is omitted, an error will result.

The available built-in scenario names are:

- `boolq`
- `imdb`
- `copyright`
- `mmlu`
- `interactive_qa_mmlu`
- `msmarco`
- `narrative_qa`
- `commonsense`
- `lsat_qa`
- `quac`
- `wikifact`
- `babi_qa`
- `real_toxicity_prompts`
- `summarization_xsum`
- `summarization_xsum_sampled`
- `summarization_cnndm`
- `truthful_qa`
- `twitter_aae`
- `disinformation`
- `gsm`
- `math`
- `natural_qa`
- `numeracy`
- `the_pile`
- `raft`
- `synthetic_efficiency`
- `synthetic_reasoning`
- `synthetic_reasoning_natural`
- `news_qa`
- `wikitext_103`
- `blimp`
- `code`
- `empatheticdialogues`
- `bold`
- `bbq`
- `civil_comments`
- `dyck_language`
- `legal_support`
- `entity_matching`
- `entity_data_imputation`
- `ice`
- `big_bench`
- `pubmed_qa`

Refer to the documentation in the corresponding `Scenario` class for each scenario's arguments, as well for descriptions of scenarios and references to the original papers and datasets.

### Run Expanders

A **`RunExpander`**, takes a RunSpec as input and outputs one or more generated RunSpecs.

After the scenario name and scenario parameters are used to construct a **base RunSpec**, the remaining parameters in the RunSpec description are used to run **`RunExpanders`** that produce **generated RunSpecs**. The generated RunSpecs (_not_ the base RunSpec) determine the runs that get executed by the HELM command-line tools.

The parameter key specifies which `RunExpander` to run. For instance, the `model` key corresponds to `ModelRunExpander`. The parameter value is passed into the `RunExpander` as an argument.

Some `RunExpanders` can result in multiple RunSpecs. For instance, if the RunSpec description contained `model=text`, `ModelRunExpander` would generate a RunSpec for each available text model.

The available built-in `RunExpander` names are:.

- `data_augmentation` 
- `global_prefix` 
- `instructions` 
- `max_tokens` 
- `max_train_instances` 
- `model` 
- `newline` 
- `num_output_tokens` 
- `num_outputs` 
- `num_prompt_tokens` 
- `num_train_trials` 
- `prompt` 
- `stop` 
- `tokenizer`

Refer to the documentation in the corresponding `RunExpander` class for the arguments and behavior of each run expander.

Generally, every RunSpec description should specify at least the `model` parameter. This determines which model(s) will be used for the run. The value of the parameter should be the name of a specific model, or the name of a category of models.

### Models

Currently, the available models are:

- `ai21/j1-grande`
- `ai21/j1-jumbo`
- `ai21/j1-large`
- `anthropic/stanford-online-all-v4-s3`
- `cohere/large-20220720`
- `cohere/medium-20220720`
- `cohere/medium-20221108`
- `cohere/small-20220720`
- `cohere/xlarge-20220609`
- `cohere/xlarge-20221108`
- `gooseai/gpt-j-6b`
- `gooseai/gpt-neo-20b`
- `huggingface/gpt-j-6b`
- `huggingface/gpt2`
- `microsoft/TNLGv2_530B`
- `microsoft/TNLGv2_7B`
- `openai/ada`
- `openai/babbage`
- `openai/code-cushman-001`
- `openai/code-davinci-001`
- `openai/code-davinci-002`
- `openai/curie`
- `openai/davinci`
- `openai/text-ada-001`
- `openai/text-babbage-001`
- `openai/text-curie-001`
- `openai/text-davinci-001`
- `openai/text-davinci-002`
- `openai/text-similarity-ada-001`
- `openai/text-similarity-babbage-001`
- `openai/text-similarity-curie-001`
- `openai/text-similarity-davinci-001`
- `together/bloom`
- `together/glm`
- `together/gpt-j-6b`
- `together/gpt-neox-20b`
- `together/opt-175b`
- `together/opt-66b`
- `together/t0pp`
- `together/t5-11b`
- `together/ul2`
- `together/yalm`