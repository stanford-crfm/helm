import importlib
import importlib.metadata
import logging
import sys
from textwrap import dedent
from helm.benchmark.run import import_user_plugins, load_entry_point_plugins


def test_import_user_plugins_accepts_modules_and_paths(tmp_path, monkeypatch):
    module_name = "temp_plugin_module"
    module_file = tmp_path / f"{module_name}.py"
    module_file.write_text("FLAG = 'from_module'\n")

    plugin_path = tmp_path / "plugin_from_path.py"
    plugin_path.write_text("FLAG = 'from_path'\n")

    monkeypatch.syspath_prepend(tmp_path)

    if module_name in sys.modules:
        importlib.invalidate_caches()
        del sys.modules[module_name]

    modules = import_user_plugins([module_name, str(plugin_path)])

    assert [module.FLAG for module in modules] == ["from_module", "from_path"]


def test_load_entry_point_plugins_handles_failures(tmp_path, monkeypatch, caplog):
    plugin_dir = tmp_path / "entrypoint_plugins"
    plugin_dir.mkdir()

    (plugin_dir / "good_plugin.py").write_text("FLAG = 'loaded'\n")
    (plugin_dir / "bad_plugin.py").write_text("raise RuntimeError('boom')\n")

    dist_info = plugin_dir / "entrypoint-0.0.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(dedent(
        """
        Metadata-Version: 2.1
        Name: entrypoint
        Version: 0.0.0
        """))
    (dist_info / "entry_points.txt").write_text(dedent(
        """
        [helm_test]
        good = good_plugin:FLAG
        bad = bad_plugin:FLAG
        """))

    monkeypatch.syspath_prepend(str(plugin_dir))
    importlib.invalidate_caches()

    caplog.set_level(logging.WARNING)
    result = load_entry_point_plugins(group="helm_test")

    assert result == ["loaded"]
    assert any("Failed to load plugin entry point" in record.message for record in caplog.records)


def test_import_user_plugins_supports_namespace_packages(tmp_path, monkeypatch):
    plugin_root = tmp_path / "plugins"
    run_specs_dir = plugin_root / "helm" / "benchmark" / "run_specs"
    run_specs_dir.mkdir(parents=True)

    (plugin_root / "helm" / "__init__.py").write_text("")
    (plugin_root / "helm" / "benchmark" / "__init__.py").write_text("")
    (run_specs_dir / "__init__.py").write_text("")

    (run_specs_dir / "custom.py").write_text(dedent(
        """
        from helm.benchmark.adaptation.adapter_spec import AdapterSpec
        from helm.benchmark.metrics.metric import MetricSpec
        from helm.benchmark.run_spec import RunSpec, run_spec_function
        from helm.benchmark.scenarios.scenario import ScenarioSpec


        @run_spec_function("custom_namespace_run")
        def build_run_spec():
            return RunSpec(
                name="custom_namespace_run",
                scenario_spec=ScenarioSpec(class_name="helm.benchmark.scenarios.scenario.Scenario"),
                adapter_spec=AdapterSpec(model="dummy"),
                metric_specs=[MetricSpec(class_name="helm.benchmark.metrics.metric.Metric")],
            )
        """))

    import helm
    import helm.benchmark
    import helm.benchmark.run_specs as run_specs_pkg

    monkeypatch.syspath_prepend(str(plugin_root))
    monkeypatch.setattr(helm, "__path__", list(helm.__path__) + [str(plugin_root / "helm")])
    monkeypatch.setattr(
        helm.benchmark,
        "__path__",
        list(helm.benchmark.__path__) + [str(plugin_root / "helm" / "benchmark")],
    )
    monkeypatch.setattr(run_specs_pkg, "__path__", list(run_specs_pkg.__path__) + [str(run_specs_dir)])
    importlib.invalidate_caches()

    import_user_plugins(["helm.benchmark.run_specs.custom"])

    from helm.benchmark.run_spec import get_run_spec_function

    assert get_run_spec_function("custom_namespace_run") is not None


def test_import_user_plugins_supports_object_spec_plugins(tmp_path, monkeypatch):
    module_name = "custom_component_plugin"
    module_file = tmp_path / f"{module_name}.py"
    module_file.write_text(dedent(
        """
        from typing import List

        from helm.benchmark.adaptation.adapter_spec import AdapterSpec
        from helm.benchmark.adaptation.request_state import RequestState
        from helm.benchmark.metrics.metric import Metric, MetricSpec
        from helm.benchmark.metrics.metric_service import MetricService
        from helm.benchmark.metrics.statistic import Stat
        from helm.benchmark.run_spec import RunSpec, run_spec_function
        from helm.benchmark.scenarios.scenario import Scenario, ScenarioMetadata, ScenarioSpec, Instance
        from helm.clients.client import Client
        from helm.common.request import Request, RequestResult
        from helm.common.tokenization_request import (
            TokenizationRequest,
            TokenizationRequestResult,
            DecodeRequest,
            DecodeRequestResult,
            TokenizationToken,
        )
        from helm.tokenizers.tokenizer import Tokenizer


        @run_spec_function("custom_plugin_run_spec")
        def build_run_spec() -> RunSpec:
            return RunSpec(
                name="custom_plugin_run_spec",
                scenario_spec=ScenarioSpec(class_name="custom_component_plugin.CustomScenario"),
                adapter_spec=AdapterSpec(model="dummy"),
                metric_specs=[MetricSpec(class_name="custom_component_plugin.CustomMetric")],
            )


        class CustomScenario(Scenario):
            name = "custom_plugin_scenario"
            description = "A custom scenario for plugin tests."
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
        """))

    monkeypatch.syspath_prepend(tmp_path)

    if module_name in sys.modules:
        importlib.invalidate_caches()
        del sys.modules[module_name]

    import_user_plugins([module_name])

    from helm.benchmark.metrics.metric import Metric, MetricSpec, create_metric
    from helm.benchmark.model_deployment_registry import ClientSpec
    from helm.benchmark.scenarios.scenario import Scenario, ScenarioSpec, create_scenario
    from helm.benchmark.run_spec import get_run_spec_function
    from helm.benchmark.tokenizer_config_registry import TokenizerSpec
    from helm.clients.client import Client
    from helm.common.object_spec import create_object
    from helm.tokenizers.tokenizer import Tokenizer

    scenario = create_scenario(ScenarioSpec(class_name=f"{module_name}.CustomScenario"))
    metric = create_metric(MetricSpec(class_name=f"{module_name}.CustomMetric"))
    client = create_object(ClientSpec(class_name=f"{module_name}.CustomClient"))
    tokenizer = create_object(TokenizerSpec(class_name=f"{module_name}.CustomTokenizer"))

    assert isinstance(scenario, Scenario)
    assert isinstance(metric, Metric)
    assert isinstance(client, Client)
    assert isinstance(tokenizer, Tokenizer)
    assert get_run_spec_function("custom_plugin_run_spec") is not None
