from .scenario import create_scenario, Scenario
from .test_utils import get_scenario_spec_tiny


class TestScenario:
    def setup_method(self, method):
        self.scenario: Scenario = create_scenario(get_scenario_spec_tiny())

    def test_str(self):
        assert (
            str(self.scenario) == "Scenario: Simple Scenario\n"
            "A simple scenario.\n"
            "Tags: simple\n"
            "4 instances\n\n"
            "------- Instance 1/4: train\n"
            "Input: 4 18 2 8 3\n"
            "Reference (correct): 8\n"
            "Reference (): -1\n\n"
            "------- Instance 2/4: train\n"
            "Input: 14 15 12 6 3\n"
            "Reference (correct): 6\n"
            "Reference (): -1\n\n"
            "------- Instance 3/4: test\n"
            "Input: 0 12 13 19 0\n"
            "Reference (correct): 19\n"
            "Reference (): -1\n\n"
            "------- Instance 4/4: test\n"
            "Input: 8 7 18 3 10\n"
            "Reference (correct): 8\n"
            "Reference (): -1"
        )
