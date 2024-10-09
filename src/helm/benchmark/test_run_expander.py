import unittest

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.run_expander import IncreaseMaxTokensRunExpander
from helm.benchmark.run_spec import RunSpec
from helm.benchmark.scenarios.scenario import ScenarioSpec


class RunExpanderTest(unittest.TestCase):
    def test_increase_max_tokens_length(self):
        # Test that each Instance is given a unique ID and is preserved through data augmentation
        run_expander = IncreaseMaxTokensRunExpander(42)

        original_run_spec = RunSpec(
            name="test:test=test",
            scenario_spec=ScenarioSpec(class_name="FakeScenario", args={}),
            adapter_spec=AdapterSpec(max_tokens=137),
            metric_specs=[],
        )
        actual_run_specs = run_expander.expand(original_run_spec)
        expected_run_specs = [
            RunSpec(
                name="test:test=test",
                scenario_spec=ScenarioSpec(class_name="FakeScenario", args={}),
                adapter_spec=AdapterSpec(max_tokens=137 + 42),
                metric_specs=[],
            )
        ]
        self.assertListEqual(actual_run_specs, expected_run_specs)
