# Importing Custom Modules

HELM is a modular framework with a plug-in architecture. You can write your own implementation for run specs, clients, tokenizers, scenarios, metrics, annotators, perturbations, and window services and use them in HELM with HELM installed as a library, without needing to modify HELM itself.

Most custom components in HELM fall into one of two categories:

1. **Run specs (decorator registration).** Run specs are registered at import time via `@run_spec_function(...)` and are discoverable by name when you invoke `helm-run`.
2. **ObjectSpec-backed classes (import by class name).** Scenarios, metrics, clients, tokenizers, annotators, perturbations, and window services are defined as classes. HELM loads them by importing the module and looking up the class specified in the `class_name` field of the relevant `ObjectSpec` (`ScenarioSpec`, `MetricSpec`, `ClientSpec`, `TokenizerSpec`, `AnnotatorSpec`, `PerturbationSpec`, `WindowServiceSpec`).

Because of this, the main way to plug in custom code is to make your modules importable and then reference them either via a run spec decorator or via a `class_name` in an `ObjectSpec`.

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

## What plugin code looks like

Below is a compact example showing both registration styles in a single module. Run specs use a decorator, while classes are loaded by name through `ObjectSpec` objects.

```python
# my_package/helm_plugin.py
from typing import List

from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.metrics.metric import Metric, MetricSpec
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.scenarios.scenario import Scenario, ScenarioSpec, ScenarioMetadata, Instance
from helm.clients.client import Client
from helm.common.request import Request, RequestResult
from helm.tokenizers.tokenizer import Tokenizer
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
    TokenizationToken,
)


@run_spec_function("custom_run_spec")
def build_custom_run_spec() -> RunSpec:
    return RunSpec(
        name="custom_run_spec",
        scenario_spec=ScenarioSpec(class_name="my_package.helm_plugin.CustomScenario"),
        adapter_spec=AdapterSpec(model="dummy"),
        metric_specs=[MetricSpec(class_name="my_package.helm_plugin.CustomMetric")],
    )


class CustomScenario(Scenario):
    name = "custom_scenario"
    description = "A tiny scenario used for testing."
    tags = ["custom"]

    def get_instances(self, output_path: str) -> List[Instance]:
        return []

    def get_metadata(self) -> ScenarioMetadata:
        return ScenarioMetadata(name=self.name, main_metric="custom_metric", main_split="test")


class CustomMetric(Metric):
    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        return []


class CustomClient(Client):
    def make_request(self, request: Request) -> RequestResult:
        return RequestResult(success=True, cached=False, embedding=[], completions=[])


class CustomTokenizer(Tokenizer):
    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        return TokenizationRequestResult(
            success=True,
            cached=False,
            text=request.text,
            tokens=[TokenizationToken(value=request.text)],
        )

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        return DecodeRequestResult(success=True, cached=False, text="".join(map(str, request.tokens)))
```

Notes:
- Run specs are registered by the decorator when the module is imported.
- Classes are loaded by `class_name` (for example, `"my_package.helm_plugin.CustomScenario"` in a `ScenarioSpec`).
- Clients and tokenizers are typically referenced from model deployments or tokenizer config entries; those config entries must be registered (for example via `register_configs_from_directory()` or `--local-path`) so the specs can be resolved.


## Adding the current working directory to PYTHONPATH

HELM will only be able to use custom classes that can be imported by Python. Depending which plugin strategy you use, you may need to do additional steps. This is not necessary with python entry points or path based explicit plugin specification.

If the custom classes live in a Python module under the current working directory, you should modify `PYTHONPATH` to make that Python module importable.

This is required because - in some environments - Python does not add the current working directory to the Python module search path when running command line commands / Python entry points such as `helm-run`. See [Python's documentation](https://docs.python.org/3/library/sys_path_init.html) for more details.

For example, suppose you implemented a custom `Client` subclass named `MyClient` in the `my_client.py` file under your current working directory, and you have a `ClientSpec` specifying the `class_name` as `my_client.MyClient`.

To make your file importable by Python, you have to add `.` to your `PYTHONPATH` so that Python will search in your current working directory for your custom Python modules.

In Bash, you can do this by running `export PYTHONPATH=".:$PYTHONPATH"` before running `helm-run`, or by prefixing `helm-run` with `PYTHONPATH=".:$PYTHONPATH"`.
