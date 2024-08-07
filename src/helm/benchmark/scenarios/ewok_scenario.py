import datasets
import os
from typing import List

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TEST_SPLIT,
    TRAIN_SPLIT,
    Scenario,
    Instance,
    Reference,
    Input,
    Output,
)
from helm.common.general import ensure_directory_exists


class EWoKScenario(Scenario):
    """Elements of World Knowledge (EWoK)

    Elements of World Knowledge (EWoK) is a framework for evaluating world modeling in
    language models by testing their ability to use knowledge of a concept to match a
    target text with a plausible/implausible context. EWoK targets specific concepts
    from multiple knowledge domains known to be vital for world modeling in humans.
    Domains range from social interactions (help/hinder) to spatial relations (left/right).
    Both, contexts and targets are minimal pairs. Objects, agents, and locations in the items
    can be flexibly filled in enabling easy generation of multiple controlled datasets.

    EWoK-CORE-1.0 is a dataset of 4,374 items covering 11 world knowledge domains."""

    name = "ewok"
    description = (
        "Elements of World Knowledge (EWoK) is a benchmark for evaluating world modeling by testing their ability to "
        "use knowledge of a concept to match a target text with a plausible/implausible context."
    )
    tags = ["world knowledge"]

    DOMAINS = [
        "agent_properties",
        "material_dynamics",
        "material_properties",
        "physical_dynamics",
        "physical_interactions",
        "physical_relations",
        "quantitative_properties",
        "social_interactions",
        "social_properties",
        "social_relations",
        "spatial_relations",
    ]

    def __init__(self, domain: str = "all"):
        super().__init__()
        if domain != "all" and domain not in EWoKScenario.DOMAINS:
            raise Exception(f"Unknown domain '{domain}', valid domains are {EWoKScenario.DOMAINS}")
        self.domain = domain.replace("_", "-")

    def get_instances(self, output_path: str) -> List[Instance]:
        next_instance_index = 0

        def generate_instance_id() -> str:
            nonlocal next_instance_index
            instance_id = f"id{next_instance_index}"
            next_instance_index += 1
            return instance_id

        cache_dir = os.path.join(output_path, "data")
        ensure_directory_exists(cache_dir)

        # TODO: Switch this to the production dataset when available.
        dataset = datasets.load_dataset(
            "ewok-core/ewok-core-1.0",
            split="test",
            revision="34d912a608066c92e2990a0328ffc3bd9a716042",
            cache_dir=cache_dir,
        )

        instances: List[Instance] = []
        for row in dataset:
            contexts = [row["Context1"], row["Context2"]]
            targets = [row["Target1"], row["Target2"]]
            # References are category ID, followed by level 2, 3 and 4 category names.
            for target_index, target in enumerate(targets):
                input = Input(text=target)
                references = [
                    Reference(output=Output(text=context), tags=[CORRECT_TAG] if context_index == target_index else [])
                    for context_index, context in enumerate(contexts)
                ]
                instance = Instance(id=generate_instance_id(), input=input, references=references, split=TEST_SPLIT)
                # Filtering by domain after generate instance IDs,
                # so that instance IDs for an item is invariant regardless of domain filtering.
                if self.domain == "all" or self.domain == row["Domain"]:
                    instances.append(instance)
        instances.extend(
            [
                Instance(
                    id=generate_instance_id(),
                    input=Input(text="I drew a ball from the bag."),
                    references=[
                        Reference(output=Output(text="The bag is full of blocks."), tags=[]),
                        Reference(output=Output(text="The bag is full of balls."), tags=[CORRECT_TAG]),
                    ],
                    split=TRAIN_SPLIT,
                ),
                Instance(
                    id=generate_instance_id(),
                    input=Input(text="The boy chose to eat a cookie."),
                    references=[
                        Reference(output=Output(text="The boy likes cookies."), tags=[CORRECT_TAG]),
                        Reference(output=Output(text="The boy does not like cookies."), tags=[]),
                    ],
                    split=TRAIN_SPLIT,
                ),
            ]
        )
        return instances
