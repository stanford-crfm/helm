from .scenario import create_scenario, Scenario
from .run_specs import get_scenario_spec_tiny, get_numeracy_scenario_spec


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


class TestNumeracyScenario:
    def setup_method(self, method):
        self.scenario: Scenario = create_scenario(get_numeracy_scenario_spec(num_in_context_samples=10))

    def test_render_lines(self):
        instances = self.scenario.get_instances()
        # print('\n'.join(self.scenario.render_lines(instances)))
        print(self.scenario.render_lines(instances))
        assert self.scenario.render_lines(instances) == [
            "Name: numeracy",
            "Description: A simple scenario",
            "Tags: [simple]",
            "4 instances",
            "",
            "------- Instance 1/4: [train]",
            "Input: 32, 29\n15, 12\n63, 60\n97, 94\n57, 54\n60, 57\n83, 80\n48, 45\n26, 23\n12,",
            "Reference ([correct]): 9",
            "",
            "------- Instance 2/4: [train]",
            "Input: 49, 143\n55, 161\n77, 227\n97, 287\n98, 290\n0, -4\n89, 263\n57, 167\n34, 98\n92,",
            "Reference ([correct]): 272",
            "",
            "------- Instance 3/4: [test]",
            "Input: 40, 37\n3, 0\n2, -1\n3, 0\n83, 80\n69, 66\n1, -2\n48, 45\n87, 84\n27,",
            "Reference ([correct]): 24",
            "",
            "------- Instance 4/4: [test]",
            "Input: 67, 197\n28, 80\n97, 287\n56, 164\n63, 185\n70, 206\n29, 83\n44, 128\n29, 83\n86,",
            "Reference ([correct]): 254",
            "",
        ]
