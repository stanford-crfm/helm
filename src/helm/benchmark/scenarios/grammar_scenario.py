from typing import List

from helm.benchmark.scenarios.scenario import Scenario, Instance, Input, TEST_SPLIT
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
