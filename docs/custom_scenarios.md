# Custom Scenarios

HELM comes with more than a hundred built-in scenarios. However, you may want to run HELM on a scenario that is not built into HELM yet, or you may want to run HELM on scenarios that use your private datasets. Because HELM is a modular framework with a plug-in architecture, you can run evaluations with your custom scenarios on HELM without needing to modify HELM code.

There are two steps to adding a custom scenario: adding the custom `Scenario` subclass, and adding a custom run spec function.

The easiest way to implement the custom `Scenario` subclass and the custom run spec function would be to copy from an appropriate example and then make the appropriate modifications. Determine the **task** your scenario, then find the corresponding example `Scenario` subclass and run spec function from the list below. You can find them in the `simple_scenarios.py` and `simple_run_specs.py` files.

- **Multiple-choice question answering**: `SimpleMCQAScenario` and `get_simple_mcqa_run_spec()`
- **Short-answer question answering**: `SimpleShortAnswerQAScenario` and `get_simple_short_answer_qa_run_spec()`
- **Open-ended question answering**: This is the same as short-answer question answering.
- **Multi-class classification**: `SimpleClassificationScenario` and `get_simple_classification_run_spec()`
- Multi-label classification: This is currently unsupported by HELM.
- **Sentiment analysis**: This a sub-type of the Classification task. Set `input_noun` to "Passage" (or something similar), set `output_noun` to "Sentiment", and include in instructions the names of the classes (e.g. "Positive" and "Negative").
- **Toxicity detection**: This a sub-type of the Classification task. Set `input_noun`, `output_noun` and `instructions` appropriately.

Note: Both the custom `Scenario` subclass and the custom run spec function will be added to custom Python modules that have to be importable by Python. The easiest way to do this is to place your custom Python modules under the current working directory and then run `export PYTHONPATH=".:$PYTHONPATH"` in your shell. Refer to the Importing Custom Modules documentation for other ways to do this.

## Custom `Scenario` subclass

The custom `Scenario` subclass should live in any importable Python module.

For this tutorial, we will create a `MyScenario` class in the the `my_scenario` module. Make a file called `./my_scenario.py` under the current working directory. Create a new class called `MyScenario`. Find the appropriate example scenario and copy its implementation into `MyScenario`, making sure to also copy the required imports.

Now we will create a test for the scenario to make sure that it is working correctly. Create a file called `./my_scenario_test.py` under the current working directory. Create a `test_my_scenario()` function in this file. Find the appropriate example scenario test from `test_simple_scenarios.py` and copy its implementation into `test_my_scenario()`.

You can now run `python3 -m pytest test_my_scenario.py` to test the example scenario. The test should pass. If you get a `ModuleNotFound` error, you should set up your `PYTHONPATH` as explained above, and then try again.

Now, modify `MyScenario` to include the actual logic to load the instances from your dataset. Use the test to ensure that your implementation is working.

### Downloading data to local disk

Frequently, you will want to download and cache data onto the local disk, rather than downloading it from the internet every time.

`get_instances(self, output_path: str)`

### Working with JSON / JSONL / CSV files

`datasets.load_dataset()`

### Working with Hugging Face datasets

## Custom run spec function

A run spec function is the entry point to the scenario. A run spec function produces a `RunSpec` (a configuration for an evaluation run). `helm-run` will run the run spec function to get the `RunSpec`, and then it will run the evaluation defined by that `RunSpec`.

HELM will search for modules with names matching these patterns for run spec functions:

- `helm.benchmark.run_specs.*_run_specs`
- `helm_*_run_specs` (i.e. a root module)

For this tutorial, we will create a `get_my_run_spec()` function in the `helm_my_run_specs` module. In your current working directory, create a file called `helm_my_run_specs.py`. Cerate a `get_my_run_spec()` function in this file. Find the appropriate example run spec function from `simple_run_specs.py` and copy its implementation into `get_my_run_spec()`. Change the 


Now run:

```
helm-run --run-entry custom:model=openai/gpt2 --suite custom --max-eval-instances 5
```

If you get a `ValueError: Unknown run spec name` error, you should set up your `PYTHONPATH` as explained above, and then try again.