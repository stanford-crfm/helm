# Importing Custom Modules

HELM is a modular framework with a plug-in architecture. You can write your own implementation for a client, tokenizer, scenario or metric and use them in HELM with HELM installed as a library, without needing to modify HELM itself.

The main way for you to use your code in HELM is to write a custom Python class that is a subclass of `Client`, `Tokenizer`, `Scenario` or `Metric` in a Python module. You can then specify a `ClientSpec`, `TokenizerSpec`, `ScenarioSpec` or `MetricSpec` (which are all classes of `ObjectSpec`) where the `class_name` is the name of your custom Python class.

## Plugin-style registration

Extensions must register themselves at import time, and HELM supports four ways to accomplish this:

1. **Python entry points (recommended).** If your custom code is organized as an installable Python package, you can declare a `helm` entry-point group in your `pyproject.toml`:

   ```toml
   [project.entry-points.helm]
   my_plugin = "my_package.helm_plugin"
   ```

   This will allow `helm-run` to automatically import your plugin to make it available at runtime. Installing your package as a wheel (or in developer mode via `pip install -e .`), ensures helm can always discover your plugin without explicit modification of `PYTHONPATH`.

2. **Namespace packages under the `helm` module.** HELM automatically discovers run specs placed in the `helm.benchmark.run_specs` namespace (via [`pkgutil.iter_modules`](https://docs.python.org/3/library/pkgutil.html#pkgutil.iter_modules)). You can ship a separate package that contributes modules to this namespace (for example, `helm/benchmark/run_specs/my_run_spec.py`) and registers additional run spec functions when imported. In this case your module must be available in the `PYTHONPATH` as described below. 

3. **Explicit imports via `--plugins`.** This option explicitly tells `helm-run` which module contains your plugin code. You can pass either importable module names or filesystem paths to Python files:

   ```bash
   helm-run --plugins my_package.helm_plugin /path/to/local_plugin.py ...
   ```

   HELM resolves module names with `importlib.import_module` and file paths with `ubelt.import_module_from_path`, so you can load quick experiments without packaging them. Paths are interpreted literally; module names still need to be importable (for example, by adjusting `PYTHONPATH` as described below).

4. **Write a Python wrapper script**. There is no need to use the `helm-run` entry point, you can instead write a Python wrapper script that calls `helm.benchmark.run.run_benchmark()`. Python will automatically add the directory containing that script to the Python module search path. If your custom classes live in a Python module under that directory, they will automatically be importable by Python. See [Python's documentation](https://docs.python.org/3/library/sys_path_init.html) for more details.

For example, suppose you implemented a custom `Client` subclass named `MyClient` in the `my_client.py` file under your current working directory, and you have a `ClientSpec` specifying the `class_name` as `my_client.MyClient`. Suppose you added a script called `run_helm.py` that calls `helm.benchmark.run.run_benchmark()` directly. When run using `python run_helm.py`, HELM will be able to import your modules without any additional changes.


## Adding the current working directory to PYTHONPATH

HELM will only be able to use custom classes that can be imported by Python. Depending which plugin strategy you use, you may need to do additional steps. This is not necessary with python entry points or path based explicit plugin specification.

If the custom classes live in a Python module under the current working directory, you should modify `PYTHONPATH` to make that Python module importable.

This is required because - in some environment - Python does not add the current working directory to the Python module search path running when using command line comments / Python entry points such as `helm-run`. See [Python's documentation](https://docs.python.org/3/library/sys_path_init.html) for more details.

For example, suppose you implemented a custom `Client` subclass named `MyClient` in the `my_client.py` file under your current working directory, and you have a `ClientSpec` specifying the `class_name` as `my_client.MyClient`.

To make your file importable by Python, you have to add `.` to your `PYTHONPATH` so that Python will search in your current working directory for your custom Python modules.

In Bash, you can do this by running `export PYTHONPATH=".:$PYTHONPATH"` before running `helm-run`, or by prefixing `helm-run` with `PYTHONPATH=".:$PYTHONPATH `.
