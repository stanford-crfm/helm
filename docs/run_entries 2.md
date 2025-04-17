# Run Entries

## Using run entries

Run entries are the main way of specifying to `helm-run` which evaluation runs to execute. For instance, in order to evaluate GPT-2 on MedQA, we would pass the following run entry to `helm-run`:

```
med_qa:model=openai/gpt2
```

There are two ways of passing the run entry to `helm-run`. We can use the `--run-entries` flag. For example:

```
helm-run --run-entries med_qa:model=openai/gpt2 --suite my-suite --max-eval-instances 10
```

Alternatively, we can put the run entry into a `run_entries.conf` file, and the pass that file to `helm-run` using the `--conf-file` flag. The `run_entries.conf` file is a **run entry configuration file** that conforms to the format documented [here](run_entries_configuration_files.md). For example:

```
helm-run --conf-file run_entries.conf --suite my-suite --max-eval-instances 10
```

## Constructing run entires

### Specifying the run spec function name

The first part of the run entry before the `:` is the run spec function name. For example, in the run entry `med_qa:model=openai/gpt2`, the run spec function name is `med_qa`.

A catalog of all run spec function names will be added to the documentation in the future. For now, the best way to find the run spec function name is to look through functions decorated with the `@run_spec_function()` in the Python modules `helm.benchmark.run_specs.*_run_specs`. The run spec function name is the decorator's parameter e.g. `@run_spec_function("med_qa")` indicates a run spec function name of `med_qa`.

Note: the run spec function name is frequently the same as the scenario name by convention, but this is not always the case. For instance, the `openbookqa` scenario has a run spec function that is named `commonsense`.

### Run entry arguments

The second part of the run entry after the `:` is a mapping of argument names to argument values. The string has the format `arg_name_1=arg_value_1,arg_name_2=arg_value_2` i.e. the name and value of each argument is joined by `=`, and the argument name-value pairs are joined by `,`. All argument values must be non-empty strings.

The run entry arguments are used for two different things: run spec function arguments, and run expanders. For instance, in the example run entry `mmlu:subject=anatomy,model=openai/gpt2`, a run spec function argument is specified by `subject=anatomy`, and a run expander is specified by `model=openai/gpt2`.

As in the above example, you can mix run expanders and run spec function arguments in a single run entry. If there is a name conflict between a run expander name and a run spec function argument name, the run expander has precedence. 

### Run spec function arguments

Some run spec functions take in arguments. For instance, the MMLU run spec function `get_mmlu_spec()` takes in a `subject` argument. MMLU is a question answering scenario that covers multiple academic subjects. The `subject` argument specifies that the question set corresponding to that academic subject should be used for that evaluation run. For instance, to evaluate MMLU with the anatomy subject on GPT-2, the run entry should be:

`mmlu:subject=anatomy,model=openai/gpt2`

A catalog of all run spec functions' parameters will be added to the documentation in the future. For now, the best way to find the run spec function parameters would be to inspect the function definition in the Python modules `helm.benchmark.run_specs.*_run_specs` for the run spec function in question.

### Run expanders

Run expanders are functions that modify how evaluation runs work. Concretely, a run expander operates on a configuration of an evaluation run (a `RunSpec`) and produces zero, one or multiple evaluation runs configurations with modified configurations (`RunSpecs`).

Run expanders are an advanced topic. For most use cases, the only run expander that you will need to use is the `model` run expander. The `model=openai/gpt2` argument pair in the run entry indicates that the evaluation run should use the `openai/gpt2` model. More explanation may be added to the documentation in the future.

### Run entry naming

The first part of the run entry name is usually be the name of the scenario by convention, but this may not always be the case. For instance, the run entry `commonsense:dataset=openbookqa,model=openai/gpt2` uses the `openbookqa` scenario.

The first part of the run entry name is usually be the name of the run spec function name by convention, but this may not always be the case. For instance, the run entry `disinformation:type=wedging,model=openai/gpt2` results in the `RunSpec` name `disinfo:type=wedging,model=openai_gpt2`.

### Run entries and `RunSpec`s

You may have noticed that some run entries can produce multiple evaluation runs. Concretely, single run entry can produce multiple `RunSpec`s, and each `RunSpec` specifies a single evaluation run.

This is because run expanders are functions that take in a `RunSpec` and can produce multiple `RunSpec`. As explained previously, the `model` run expander is an example of this.

### The `model` run expander

The `model` run expander is the most commonly used run expander. As discussed earlier, it can be used to set the model for each run entry.

The `model` run expander also supports **wildcard values**. For instance, the `med_qa:model=text` run entry will run the `med_qa` scenario on _every_ text model that `helm-run` can find in its configuration files. The wildcard is intended to be used in conjuction with the `--models-to-run`, which controls which models will actually be evaluated. For example, `helm-run --run-entries med_qa:model=text --models-to-run openai/gpt2 openai/gpt-3.5-turbo-613` will run `med_qa` on _only_ `openai/gpt2` and `openai/gpt-3.5-turbo-613`.

Wildcard values for the `model` run expander are common used in **run entries configuration files** which will are discussed [here](run_entries_configuration_files.md).
