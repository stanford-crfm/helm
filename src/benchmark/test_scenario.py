from .scenario import create_scenario, Scenario
from .test_utils import get_scenario_spec_tiny


class TestScenario:
    def setup_method(self, method):
        self.scenario: Scenario = create_scenario(get_scenario_spec_tiny())

    def test_render_lines(self):
        instances = self.scenario.get_instances()
        assert self.scenario.render_lines(instances) == [
            "Name: simple1",
            "Description: A simple scenario",
            "Tags: [simple]",
            "4 instances",
            "",
            "------- Instance 1/4: [train]",
            "Input: 4 18 2 8 3",
            "Reference ([correct]): 8",
            "Reference ([]): -1",
            "",
            "------- Instance 2/4: [train]",
            "Input: 14 15 12 6 3",
            "Reference ([correct]): 6",
            "Reference ([]): -1",
            "",
            "------- Instance 3/4: [test]",
            "Input: 0 12 13 19 0",
            "Reference ([correct]): 19",
            "Reference ([]): -1",
            "",
            "------- Instance 4/4: [test]",
            "Input: 8 7 18 3 10",
            "Reference ([correct]): 8",
            "Reference ([]): -1",
            "",
        ]
