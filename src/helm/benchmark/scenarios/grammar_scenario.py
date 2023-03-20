from typing import List

from .scenario import Scenario, Instance, Input, TEST_SPLIT
from .grammar import read_grammar, generate_derivations, Derivation, get_values


class GrammarScenario(Scenario):
    """
    A scenario whose instances are generated from a grammar.
    """

    name = "grammar"
    description = "Scenarios based on a grammar"

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def get_instances(self) -> List[Instance]:
        # Generate derivations
        grammar = read_grammar(self.path)
        derivations = generate_derivations(grammar)

        def derivation_to_instance(derivation: Derivation) -> Instance:
            return Instance(
                input=Input(text="".join(get_values(derivation))),
                references=[],
                split=TEST_SPLIT,
            )

        # Build the instances
        instances: List[Instance] = list(map(derivation_to_instance, derivations))
        return instances
