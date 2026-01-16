# Importing Custom Modules

HELM is a modular framework with a plug-in architecture. You can write your own implementation for run specs, clients, tokenizers, scenarios, or metrics and use them in HELM with HELM installed as a library, without needing to modify HELM itself.

In this document, a **plugin** means **user-provided Python code that extends HELM**. Practically, a plugin is a Python *module* that either:

- defines classes that HELM can load by fully-qualified name (e.g., `my_pkg.my_metric.CustomMetric`), and/or
- registers run specs when the module is imported (via a decorator).

This guide explains:

- how Python importability affects HELM
- how HELM discovers custom code
- your options for loading plugin modules
- a complete end-to-end example using Python entry points (recommended)

---

## Making your code importable (Python basics)

HELM will only be able to use custom code that can be imported by Python. In this guide, there are two main ways to make your code importable:

1. **Install it as a Python package** (optionally in editable mode).
2. **Add a directory to `PYTHONPATH`** so Python searches it for modules.

### Add the current working directory to PYTHONPATH

If the custom code lives in a Python module under the current working directory, you may need to modify `PYTHONPATH` to make that module importable.

This is required because Python does not add the current working directory to the Python module search path when using command line commands / Python entry points such as `helm-run`. See [Python's documentation](https://docs.python.org/3/library/sys_path_init.html) for more details.

For example, suppose you implemented a custom `Client` subclass named `MyClient` in the `my_client.py` file under your current working directory, and you have a `ClientSpec` specifying the `class_name` as `my_client.MyClient`.

To make your file importable by Python, you have to add `.` to your `PYTHONPATH` so that Python will search in your current working directory for your custom Python modules.

In Bash, you can do this by running `export PYTHONPATH=".:$PYTHONPATH"` before running `helm-run`, or by prefixing `helm-run` with `PYTHONPATH=".:$PYTHONPATH"`.

### Put your custom code in a Python package

If your custom code is located in a Python package, you can simply install your package (optionally in editable mode) and it will automatically be importable by Python. Be sure to install your Python package in the same Python environment as HELM.

### Write a Python wrapper script

If you are using a Python wrapper script that calls `helm.benchmark.run.run_benchmark()` instead of using `helm-run`, Python will automatically add the directory containing that script to the Python module search path. If your custom code lives in a Python module under that directory, it will automatically be importable by Python. See [Python's documentation](https://docs.python.org/3/library/sys_path_init.html) for more details.

---

## How HELM finds your code

Custom extensions generally work in one of two ways:

1. **ObjectSpec-backed classes (loaded by class name).**
   Clients, tokenizers, scenarios, and metrics are defined as classes. HELM loads them by:
   - importing the module portion of your fully qualified name, and then
   - looking up the class name you specify in the relevant `ObjectSpec` (e.g., `ScenarioSpec`, `MetricSpec`, `ClientSpec`, `TokenizerSpec`).

   **Key idea:** these modules only need to be *importable* by Python. They do not need to be imported ahead of time.

2. **Run specs (registered by decorator).**
   Run specs are registered *at import time* via `@helm.benchmark.run_spec.run_spec_function(...)` and are discoverable by name when you invoke `helm-run`.

   **Key idea:** the module containing the run spec function must be imported so registration code runs. Only modules that define run spec functions need to be imported for discovery.

---

## Ways to load plugin modules

The approaches below are mostly about ensuring that modules containing run spec functions get imported (so run specs register). They may also be useful for importing other code early (for example, to fail fast on syntax errors), but only run specs *require* this.

### 1) Python entry points (recommended for reusable plugins)

If your custom code is an installable Python package, declare a `helm` entry-point group in your `pyproject.toml`:

```toml
[project.entry-points.helm]
my_plugin = "my_package.helm_plugin"
```

When your package is installed (e.g., as a wheel or with `pip install -e .`), `helm-run` can automatically import the entry point module.

With this method, `project.entry-points.helm` only needs to include modules that contain run spec functions (and any other modules you explicitly want imported up front).

### 2) Explicit imports via `--plugins` (best for quick experiments)

You can explicitly tell `helm-run` what to import. Each `--plugins` argument can be either an importable module name or a filesystem path to a `.py` file.

Importable module names (modules must already be importable, e.g., installed or on `PYTHONPATH`):

```bash
helm-run --plugins my_package.helm_plugin_a my_package.helm_plugin_b ...
```

Filesystem paths (loaded from the given `.py` files):

```bash
helm-run --plugins /path/to/local_plugin_a.py /path/to/local_plugin_b.py ...
```

How file paths work: HELM loads each `.py` file as a module. In general, the directory containing the file determines what sibling modules can be imported from that file. If your plugin file imports other local modules, ensure those modules are importable (for example, place them next to the plugin file or set `PYTHONPATH`).

### 3) Namespace packages under `helm.benchmark.run_specs` (legacy name-based method)

HELM automatically discovers run specs placed in the `helm.benchmark.run_specs` namespace (via [`pkgutil.iter_modules`](https://docs.python.org/3/library/pkgutil.html#pkgutil.iter_modules)).

You can ship a separate package that contributes modules to this namespace (for example, `helm/benchmark/run_specs/my_run_spec.py`) and registers additional run spec functions when imported.

This method requires that your package is importable (typically by installing it, or by ensuring it is on `PYTHONPATH`).

### 4) A Python wrapper script (when you don't want to use `helm-run`)

There is no need to use the `helm-run` entry point. You can instead write a Python wrapper script that calls `helm.benchmark.run.run_benchmark()`.

When you run `python your_script.py`, Python automatically adds the script's directory to the module search path. This implicitly changes import behavior in the same way as adding that directory to `PYTHONPATH`.

---

## Example plugin (entry points + run spec + ObjectSpec classes)

This compact example shows both mechanisms:

- a run spec registered via `@helm.benchmark.run_spec.run_spec_function(...)`
- a scenario and metric referenced via `class_name=...` in `ScenarioSpec`/`MetricSpec`

We use the entry point approach because it's the most robust for repeated runs.

Note: For a small tutorial, we put the run spec and the classes in one file. In larger projects you may prefer to keep your run specs in a dedicated module (the only part that must be imported up front), and keep scenarios/metrics in separate modules.

### Prerequisites

- A compatible Python (this example uses 3.11)
- [`uv`](https://docs.astral.sh/uv/) installed

### Step 1 - Initialize a packaged project

From the directory where you want the plugin project:

```bash
uv init --package my_example_helm_module --python=3.11
cd my_example_helm_module
```

### Step 2 - Create your plugin module

Create a module for your plugin code:

```bash
mkdir -p src/my_example_helm_module
touch src/my_example_helm_module/my_submodule_plugin_code.py
```

Your directory should look like:

```text
my_example_helm_module
├── pyproject.toml
├── README.md
└── src
    └── my_example_helm_module
        ├── __init__.py
        └── my_submodule_plugin_code.py
```

Write the following into `src/my_example_helm_module/my_submodule_plugin_code.py`:

```python
from typing import List, Optional

from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.metrics.metric import Metric, MetricSpec
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.scenarios.scenario import Scenario, ScenarioSpec, ScenarioMetadata, Instance
from helm.benchmark.metrics.evaluate_reference_metrics import compute_reference_metrics
from helm.benchmark.scenarios.scenario import TRAIN_SPLIT, TEST_SPLIT, CORRECT_TAG
from helm.benchmark.scenarios.scenario import Input, Output, Reference


class CustomScenario(Scenario):
    name = "custom_scenario"
    description = "A tiny scenario used for testing."
    tags = ["custom"]

    def get_instances(self, output_path: str) -> List[Instance]:
        # We include 5 TRAIN_SPLIT instances because the generation adapter
        # uses a few-shot train instances pool by default. If you return 0
        # train instances, you'll see: "only 0 training instances, wanted 5".
        examples = [
            # (question, answer, split)
            ("1+1=?", "2", TRAIN_SPLIT),
            ("2+2=?", "4", TRAIN_SPLIT),
            ("3+3=?", "6", TRAIN_SPLIT),
            ("4+4=?", "8", TRAIN_SPLIT),
            ("5+5=?", "10", TRAIN_SPLIT),
            ("6+6=?", "12", TEST_SPLIT),
            ("7+7=?", "14", TEST_SPLIT),
        ]

        instances: List[Instance] = []
        train_i = 0
        test_i = 0

        for q, a, split in examples:
            if split == TRAIN_SPLIT:
                train_i += 1
                instance_id = f"train-{train_i:03d}"
            else:
                test_i += 1
                instance_id = f"test-{test_i:03d}"

            instances.append(
                Instance(
                    id=instance_id,
                    input=Input(text=f"Q: {q}\nA:"),
                    references=[Reference(output=Output(text=a), tags=[CORRECT_TAG])],
                    split=split,
                )
            )

        return instances

    def get_metadata(self) -> ScenarioMetadata:
        return ScenarioMetadata(name=self.name, main_metric="custom_metric", main_split="test")


class CustomMetric(Metric):
    """A simple, extensible metric.

    To keep the example compact, we just call HELM's reference-metric helper.
    """

    def __init__(self, names: Optional[List[str]] = None):
        self.names = names or ["exact_match"]

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        return compute_reference_metrics(
            names=self.names,
            adapter_spec=adapter_spec,
            request_state=request_state,
            metric_service=metric_service,
        )


@run_spec_function("my_custom_run_spec")
def build_custom_run_spec() -> RunSpec:
    return RunSpec(
        name="my_custom_run_spec",
        scenario_spec=ScenarioSpec(class_name="my_example_helm_module.my_submodule_plugin_code.CustomScenario"),
        adapter_spec=AdapterSpec(method="generation"),
        metric_specs=[MetricSpec(class_name="my_example_helm_module.my_submodule_plugin_code.CustomMetric")],
    )
```

Two things to notice:

- The run spec is registered by the decorator when the module is imported.
- The scenario and metric are referenced via fully qualified `class_name` strings.

### Step 3 - Register the plugin via entry points

Edit `pyproject.toml` and add:

```toml
[project.entry-points.helm]
my_helm_plugin = "my_example_helm_module.my_submodule_plugin_code"
```

Then install your package in editable mode:

```bash
uv pip install -e .
```

### Step 4 - Run with your custom run spec

Now `helm-run` should discover your plugin through the entry point:

```bash
helm-run --run-entries my_custom_run_spec:model=openai/gpt2 --suite tutorial --max-eval-instances 10
```
