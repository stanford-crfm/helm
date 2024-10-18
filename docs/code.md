# Code Structure

**Warning** &mdash; The document is stale and was last modified more than ten months ago. The information below may be outdated and incorrect. Please proceed with caution!

## Birds-Eye View

Here's a birds-eye view of how the benchmarking process interacts with the main
classes (see `benchmark`):

- A `Scenario` (given by a `ScenarioSpec`) specifies a task and a data
  distribution.  It specifies a set of `Instance`s, where each `Instance` has
  an input (e.g., question) and a set of `Reference` outputs (e.g., multiple
  choice answers).

- A `DataPreprocessor` takes in a `Scenario` and produces a list of `Instance`s.
  Each `Instance` is given a unique ID. The set of `Instance`s is augmented
  according to `DataAugmenterSpec`.

- An `Adapter` (given by an `AdaptationSpec`) takes a list of `Instance`s and
  adapts it to a set of `Request`s to the API (e.g., the model, temperature,
  number of in-context training examples).  Formally, the output
  is a `ScenarioState` containing a set of `RequestState`s, where each
  `RequestState` consists of a `Request` and any metadata used to track the
  role of this `Request` (e.g., the relevant `Instance` and `Reference`).

- An `Executor` (given by an `ExecutionSpec`) executes each `Request` in the
  `RequestState` to produce a `RequestResult` for each one; everything is
  encapsulated in a `ScenarioState`.

- A `Metric` (given by a `MetricSpec`) takes a `ScenarioState` containing
  `RequestResults`s and produces a set of `Stat`s (e.g., accuracy, accuracy@5,
  toxicity, bias, etc.).

- A `Runner` is the top-level controller that runs the above steps and is
  driven by a set of `RunSpec`s.

There are three types of classes:

- Specifications (e.g., `AdapterSpec`, `ExecutionSpec`, `RunSpec`):
  specified manually by the user.  Note that `Scenario` and `Metric` are
  subclassed, so they are constructed by `ObjectSpec`, which specifies the
  subclass name and a free-form dictionary of arguments.
- States (e.g., `Instance`, `ScenarioState`, `Request`, `RequestResult`): these
  are automatically generated and can be serialized.
- Controllers (e.g., `Scenario`, `Adapter`, `Executor`, `Metric`, `Runner`):
  these have the bulk of the code and should not be serialized.

## Adding new scenarios

In order to implement new scenarios:

1. Create a new Python file in the `scenarios` folder.
1. Within the scenario file, create a `Scenario` class, e.g. `YourScenario`.
1. `YourScenario` should implement `get_instances`, a method that downloads the 
   dataset files if they don't already exist and returns a list of `Instance`s. 
   Each `Instance` must have a list of (potentially one)
   `Reference` answers: a correct answer may be indicated with a `CORRECT_TAG` in 
   a `Reference` instance's `tags` argument. In addition, you 
   must specify the `split` of the `Instance` as one of `TRAIN_SPLIT`,
   `VALID_SPLIT`, or `TEST_SPLIT` constants as in `scenario.py`.
   1. For `Scenario`s with datasets that cannot be publicly shared, place a copy of the
      dataset at path `restricted/<Name of the Scenario>` and read from that path.
      See `NewsQAScenario` and `ICEScenario` for some examples.
1. Note that you need not enumerate every possible correct answer (nor must
   there even necessarily be a correct answer). 
1. Make sure to document your scenario well with a clear docstring. 
1. In addition, specify its `name`, `description`, and `tags`.
1. Identify the appropriate metric for your task in one of the `*_metrics.py` files.
   If the metric you'd like to use does not exist, follow the directions in [Adding new metrics](#adding-new-metrics).
   Many will be in `basic_metrics.py`.
1. Define a function in `run_specs.py` annotated with `run_spec_function` to:
   1. Construct a `ScenarioSpec` 
   for your scenario using a class name corresponding to the Python path of 
   the class (e.g. `helm.benchmark.scenarios.your_scenario.YourScenario`) and any 
   arguments which must be passed as a dictionary of `args`.
   1. Construct an `AdapterSpec` for your
   scenario specifying the type of language model generation which must be 
   performed for the task.
   1. Construct one or more `MetricSpec`
   objects for your task, specifying the classname with the Python path of
   the object, with the same arguments as the `ScenarioSpec` constructor.
   1. Construct and return `RunSpec` object, with a 
   `name` corresponding to the scenario name and any patterns to match in 
   curly braces, a `scenario_spec`, an `adapter_spec`, `metric_specs`, 
   and `groups`. 
1. Attempt to run your task with
   `venv/bin/helm-run -r yourscenarioname:arg=value` where 
   `yourscenarioname` matches the `name` specified in YourScenario
1. Update `src/helm/benchmark/static/contamination.yaml` with models that were trained on your scenario (i.e. contaminated).
1. Add a schema to `src/helm/benchmark/static/schema.yaml` and add the scenario to `subgroups` as needed.

## Adding new metrics

To add a new metric, first determine if your metric is generic and likely to be widely used, or specific to your task.

*  For generic metrics:
   1. Add a method to `basic_metrics.py` which takes two arguments: the `gold` answer and the model's `pred`iction.
   1. Add your method to the `metric_fn_mapping` lookup.
*  For task specific metrics:
   1. Create a new `yourtask_metrics.py` file for class `YourTaskMetric` 
   which inherits from `Metric` in `metric.py`.
   1. Define methods `__init__` and `evaluate_generation` returning a list of `Stat` objects.

Your metric is responsible for producing `Stat` objects:

*  Each `Stat` should correspond to a distinct aggregate measurement over the generated examples. 
   Some may have one metric (e.g. accuracy), while others may quantify multiple aspects
   (e.g. multiple distance metrics). 
*  For each `value` generated for a `Stat`, add it to `yourstat` using `yourstat.add(value)`. 
   Usually, there will only be one value for each `Stat`, but multiple can be used, e.g. to show variance.

## Data augmentations

To apply data augmentation, create a `DataAugmenterSpec` with a list of
`PerturbationSpec`s and pass it into `RunSpec`. The following is an
example:

```python
    data_augmenter_spec = DataAugmenterSpec(
        perturbation_specs=[
            PerturbationSpec(
                class_name="helm.benchmark.augmentations.perturbation.ExtraSpacePerturbation",
                args={"num_spaces": 5},
            )
        ],
        should_perturb_references=False,
        should_augment_train_instances=False,
        should_include_original_train=False,
        should_augment_eval_instances=True,
        should_include_original_eval=True,
    )
    run_spec = RunSpec(
        ...
        data_augmenter_spec=data_augmenter_spec
    )
```

In the example above, the `DataPreprocessor` will augment the set of evaluation instances by perturbing
the original set of instances with the `ExtraSpacePerturbation`, where spaces in the text are
replaced with `num_spaces` number of spaces.

We currently only support applying a single perturbation to an instance instead of chaining
multiple perturbations and applying it onto a single instance.

### Adding a new perturbation

1. To add a new perturbation to the framework, create a new file at `src/helm/benchmark/augmentations` with the name
   `<Name of perturbation>_perturbation.py` e.g., `typo_perturbation.py`. Inside the file, create a new class
   (name it `<Name of the perturbation>Perturbation` e.g., `TypoPerturbation`)
   that extends the abstract class `Perturbation` and implement the `perturb` method which
   takes in text and outputs the perturbed text.
2. Add a test for the new perturbation in `test_perturbation.py`.

## Supporting new Hugging Face tokenizers

1. Give the tokenizer a name. Use the same name that's used in Hugging Face (e.g., "EleutherAI/gpt-j-6B").
2. In `HuggingFaceTokenizers`, we load and cache tokenizers in memory. Add logic to handle
   the tokenizer in the `load_tokenizer` method.
3. Add a test in `test_huggingface_tokenizer.py` to make sure we can load the tokenizer from Hugging Face.
4. Add a new class `<Name of tokenizer>WindowService` in file `<Name of tokenizer>_window_service.py`.
   Follow what we did for `GPTJWindowService`.
5. Import the new `WindowService` and map the model(s) to it in `WindowServiceFactory`.


## HEIM (text-to-image evaluation)

The overall code structure is the same as HELM's.

When adding new scenarios and metrics for image generation, place the Python files under the `image_generation` package 
(e.g., `src/helm/benchmark/scenarios/image_generation`).
