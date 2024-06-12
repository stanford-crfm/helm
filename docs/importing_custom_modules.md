# Importing Custom Modules

HELM is a modular framework with a plug-in architecture. You can write your own implementation for a client, tokenizer, scenario or metric and use them in HELM with HELM installed as a library, without needing to modify HELM itself.

The main way for you to use your code in HELM is to write a custom Python class that is a subclass of `Client`, `Tokenizer`, `Scenario` or `Metric` in a Python module. You can then specify a `ClientSpec`, `TokenizerSpec`, `ScenarioSpec` or `MetricSpec` (which are all classes of `ObjectSpec`) where the `class_name` is the name of your custom Python class.

However, HELM will only be able to use custom classes that can be imported by Python. Depending on your setup, you may need to do additional steps.

## Add the current working directory to PYTHONPATH

If the custom classes live in a Python module under the current working directory, you should modify `PYTHONPATH` to make that Python module importable.

This is required because Python does not add the current working directory to the Python module search path running when using command line comments / Python entry points such as `helm-run`. See [Python's documentation](https://docs.python.org/3/library/sys_path_init.html) for more details.

For example, suppose you implemented a custom `Client` subclass named `MyClient` in the `my_client.py` file under your current working directory, and you have a `ClientSpec` specifying the `class_name` as `my_client.MyClient`.

To make your file importable by Python, you have to add `.` to your `PYTHONPATH` so that Python will search in your current working directory for your custom Python modules.

In Bash, you can do this by running `export PYTHONPATH=".:$PYTHONPATH"` before running `helm-run`, or by prefixing `helm-run` with `PYTHONPATH=".:$PYTHONPATH `.

## Put your custom classes in a Python package

If your custom classes are located in a Python package, you can simply install your package (optionally in editable mode) and it will automatically be importable by Python. Be sure to install your Python package in the same Python environment as HELM.

## Write a Python wrapper script

If you are using a Python wrapper script that calls `helm.benchmark.run.run_benchmark()` instead of using `helm-run`, Python will automatically add the directory containing that script to the Python module search path. If your custom classes live in a Python module under that directory, they will automatically be importable by Python. See [Python's documentation](https://docs.python.org/3/library/sys_path_init.html) for more details.

For example, suppose you implemented a custom `Client` subclass named `MyClient` in the `my_client.py` file under your current working directory, and you have a `ClientSpec` specifying the `class_name` as `my_client.MyClient`. Suppose you added a script called `run_helm.py` that calls `helm.benchmark.run.run_benchmark()` directly. When run using `python run_helm.py`, HELM will be able to import your modules without any additional changes.
