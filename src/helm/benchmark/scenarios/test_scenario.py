from helm.benchmark.scenarios.scenario import ScenarioSpec, create_scenario, Scenario, Input, PassageQuestionInput


class TestScenario:
    def setup_method(self, method):
        scenario_spec: ScenarioSpec = ScenarioSpec(
            class_name="helm.benchmark.scenarios.simple_scenarios.Simple1Scenario",
            args={"num_input_tokens": 5, "vocab_size": 20, "num_train_instances": 2, "num_test_instances": 2},
        )
        self.scenario: Scenario = create_scenario(scenario_spec)

    def test_render_lines(self):
        instances = self.scenario.get_instances(output_path="")
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

    def test_definition_path(self):
        assert (
            self.scenario.definition_path
            == "https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/scenarios/simple_scenarios.py"
        )


def test_input_equality():
    input1 = Input(text="input1")
    assert input1 == input1
    assert input1 == Input(text="input1")
    assert input1 != Input(text="input2")


def test_passage_question_input():
    assert (
        PassageQuestionInput(passage="Short passage", question="question?").text == "Short passage\nQuestion: question?"
    )
