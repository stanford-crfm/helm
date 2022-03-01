from .scenario import create_scenario, Scenario
from .run_specs import get_scenario_spec_tiny


class TestScenario:
    def setup_method(self, method):
        self.scenario: Scenario = create_scenario(get_scenario_spec_tiny())

    def test_render_lines(self):
        instances = self.scenario.get_instances()
        assert self.scenario.render_lines(instances) == [
            "name: simple1",
            "description: A simple scenario",
            "tags: [simple]",
            "",
            "instance 0 (4 total) |train| {",
            '  input: "4 18 2 8 3"',
            '  reference [correct]: "8"',
            '  reference []: "-1"',
            "}",
            "instance 1 (4 total) |train| {",
            '  input: "14 15 12 6 3"',
            '  reference [correct]: "6"',
            '  reference []: "-1"',
            "}",
            "instance 2 (4 total) |test| {",
            '  input: "0 12 13 19 0"',
            '  reference [correct]: "19"',
            '  reference []: "-1"',
            "}",
            "instance 3 (4 total) |test| {",
            '  input: "8 7 18 3 10"',
            '  reference [correct]: "8"',
            '  reference []: "-1"',
            "}",
        ]
