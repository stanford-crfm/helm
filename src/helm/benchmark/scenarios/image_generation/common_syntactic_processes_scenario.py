from typing import List, Dict

from helm.benchmark.scenarios.scenario import Scenario, Instance, Input, TEST_SPLIT


class CommonSyntacticProcessesScenario(Scenario):
    """
    From "DALL-E 2 Fails to Reliably Capture Common Syntactic Processes", DALL-E performs poorly
    when given prompts from 8 different grammatical phenomena:

    1. Binding principles and coreference
    2. Passives
    3. Word order
    4. Coordination
    5. Comparatives
    6. Negation
    7. Ellipsis
    8. Structural ambiguity

    The benchmark has 5 examples per grammatical phenomenon (see the full list below), where
    each example can have multiple prompts. The authors generated 4 images per prompt.

    Paper: https://arxiv.org/abs/2210.12889
    """

    BINDING_PRINCIPLES: str = "binding_principles"
    PASSIVES: str = "passives"
    WORD_ORDER: str = "word_order"
    COORDINATION: str = "coordination"
    COMPARATIVES: str = "comparatives"
    NEGATION: str = "negation"
    ELLIPSIS: str = "ellipsis"
    STRUCTURAL_AMBIGUITY: str = "ambiguity"

    # All prompts and example outputs are available in Table 1 of the appendix
    PROMPT_TO_PHENOMENON: Dict[str, str] = {
        "The man paints a picture of him": BINDING_PRINCIPLES,  # 1
        "The man paints a picture of himself": BINDING_PRINCIPLES,  # 1
        "The woman paints a portrait of her": BINDING_PRINCIPLES,  # 2
        "The woman paints a portrait of herself": BINDING_PRINCIPLES,  # 2
        "The boy looks at a picture of him": BINDING_PRINCIPLES,  # 3
        "The boy looks at a picture of himself": BINDING_PRINCIPLES,  # 3
        "The young lady looks at a picture of her": BINDING_PRINCIPLES,  # 4
        "The young lady looks at a picture of herself": BINDING_PRINCIPLES,  # 4
        "The man takes a picture of him": BINDING_PRINCIPLES,  # 5
        "The man takes a picture of himself": BINDING_PRINCIPLES,  # 5
        "The woman broke the vase": PASSIVES,  # 6
        "The vase was broken by the woman": PASSIVES,  # 6
        "The plate was broken by the woman": PASSIVES,  # 7
        "The glass was broken by the man": PASSIVES,  # 8
        "The jar was broken by the man": PASSIVES,  # 9
        "The flowerpot was broken by the man": PASSIVES,  # 10
        "The dog is chasing the man": WORD_ORDER,  # 11
        "The man is chasing the dog": WORD_ORDER,  # 11
        "The man gave the letter to the woman": WORD_ORDER,  # 12
        "The man gave the woman the letter": WORD_ORDER,  # 12
        "The man is watering the plant": WORD_ORDER,  # 13
        "The plant is watering the man": WORD_ORDER,  # 13
        "The mother combs the boy": WORD_ORDER,  # 14
        "The boy combs the mother": WORD_ORDER,  # 14
        "The man gave the comb to the woman": WORD_ORDER,  # 15
        "The man gave the woman the comb": WORD_ORDER,  # 15
        "The man is drinking water and the woman is drinking orange juice": COORDINATION,  # 16
        "The woman is eating red apple and the man is eating a green apple": COORDINATION,  # 17
        "The cat is wearing two red socks and the dog is wearing one red sock": COORDINATION,  # 18
        "The boy wears a red hat and the girl wears a blue tie": COORDINATION,  # 19
        "The woman is washing the dishes and the man is washing the floor": COORDINATION,  # 20
        "The bowl has more cucumbers than strawberries": COMPARATIVES,  # 21
        "The bowl has fewer strawberries than cucumbers": COMPARATIVES,  # 22
        "The plate has more peas than carrots": COMPARATIVES,  # 23
        "The plate has fewer carrots than peas": COMPARATIVES,  # 24
        "The plate has more than seven eggs": COMPARATIVES,  # 25
        "A tall woman without a handbag": NEGATION,  # 26
        "A man with a red sweater and blue sweater and he is not wearing the former": NEGATION,  # 27
        "A rainy street without cars": NEGATION,  # 28
        "A boy with a green t-shirt without red buttons": NEGATION,  # 29
        "A tall tree not green or black": NEGATION,  # 30
        "The man is eating a sandwich and the woman an apple": ELLIPSIS,  # 31
        "The man eats pizza but the woman does not": ELLIPSIS,  # 32
        "The girl starts a sandwich and the boy a book": ELLIPSIS,  # 33
        "The man drinks water and the woman orange juice": ELLIPSIS,  # 34
        "The woman wears a blue shirt, but the man does not": ELLIPSIS,  # 35
        "The man saw the boy in his car": STRUCTURAL_AMBIGUITY,  # 36
        "The man saw the lion with the binoculars": STRUCTURAL_AMBIGUITY,  # 37
        "The boy saw the girl using a magnifying glass": STRUCTURAL_AMBIGUITY,  # 38
        "There are three boys and each is wearing a hat": STRUCTURAL_AMBIGUITY,  # 39
        "Two cars painted a different color": STRUCTURAL_AMBIGUITY,  # 40
        "Two cars each painted a different color": STRUCTURAL_AMBIGUITY,  # 40
    }

    name = "common_syntactic_processes"
    description = "Prompts from 8 different grammatical phenomena ([paper](https://arxiv.org/abs/2210.12889))."
    tags = ["text-to-image"]

    def __init__(self, phenomenon: str):
        super().__init__()
        self.phenomenon: str = phenomenon

    def get_instances(self, _) -> List[Instance]:
        return [
            # There are no reference images
            Instance(Input(text=prompt), references=[], split=TEST_SPLIT)
            for prompt, phenomenon in self.PROMPT_TO_PHENOMENON.items()
            if phenomenon == self.phenomenon
        ]
