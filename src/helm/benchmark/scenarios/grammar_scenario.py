from typing import List

from helm.benchmark.presentation.taxonomy_info import TaxonomyInfo
from helm.benchmark.scenarios.scenario import Scenario, Instance, Input, TEST_SPLIT, ScenarioMetadata
from helm.benchmark.scenarios.grammar import read_grammar, generate_derivations, Derivation, get_values, get_tags


class GrammarScenario(Scenario):
    """
    A scenario whose instances are generated from a grammar (see `grammar.py`).
    """

    name = "grammar"
    description = "Scenarios based on a grammar"
    tags = ["instructions"]

    def __init__(self, path: str, tags: str = ""):
        super().__init__()
        self.path = path
        self.tags = tags.split(",") if tags else []

    def get_instances(self, output_path: str) -> List[Instance]:
        # Generate derivations
        grammar = read_grammar(self.path)
        derivations = generate_derivations(grammar)

        # Keep only the derivations that match all of `self.tags`
        def keep_derivation(derivation: Derivation) -> bool:
            tags = get_tags(derivation)
            return all(tag in tags for tag in self.tags)

        derivations = list(filter(keep_derivation, derivations))

        def derivation_to_instance(derivation: Derivation) -> Instance:
            return Instance(
                input=Input(text="".join(get_values(derivation))),
                references=[],
                split=TEST_SPLIT,
            )

        # Build the instances from those derivations
        instances: List[Instance] = list(map(derivation_to_instance, derivations))

        return instances

    def get_metadata(self) -> ScenarioMetadata:
        return ScenarioMetadata(
            name="grammar",
            display_name="Best ChatGPT Prompts",
            short_display_name="Best ChatGPT Prompts",
            description="A list of “best ChatGPT prompts to power your workflow” summarized by "
            "[GRIDFITI](https://gridfiti.com/best-chatgpt-prompts/).",
            taxonomy=TaxonomyInfo(
                task="open-ended instruction following",
                what="Instructions for LLMs",
                when="2023",
                who="Gridfiti Staff",
                language="English",
            ),
            main_metric="Helpfulness",
            main_split="test",
        )
