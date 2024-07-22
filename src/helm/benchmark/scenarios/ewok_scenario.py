import datasets
import dataclasses
import os
import random
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
    """Elements of World Knowledge (EWOK)

    Elements of World Knowledge (EWOK) is a framework for evaluating world modeling in
    language models by testing their ability to use knowledge of a concept to match a
    target text with a plausible/implausible context. EWOK targets specific concepts
    from multiple knowledge domains known to be vital for world modeling in humans.
    Domains range from social interactions (help/hinder) to spatial relations (left/right).
    Both, contexts and targets are minimal pairs. Objects, agents, and locations in the items
    can be flexibly filled in enabling easy generation of multiple controlled datasets.

    EWOK-CORE-1.0 is a dataset of 4,374 items covering 11 world knowledge domains."""

    name = "ewok"
    description = (
        "EWOK is a benchmark for evaluating world modeling by testing their ability to "
        "use knowledge of a concept to match a target text with a plausible/implausible context."
    )
    tags = ["world knowledge"]

    def get_instances(self, output_path: str) -> List[Instance]:
        cache_dir = os.path.join(output_path, "data")
        ensure_directory_exists(cache_dir)

        # TODO: Switch this to the production dataset when available.
        dataset = datasets.load_dataset("ewok-core/ewok-core-1.0", split="test", cache_dir=cache_dir)
        instances: List[Instance] = []
        # TODO: Allow users to filter by category
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
                instance = Instance(input=input, references=references, split=TEST_SPLIT)
                instances.append(instance)
        train_indexes = random.sample(list(range(len(instances))), k=10)
        for train_index in train_indexes:
            instances[train_index] = dataclasses.replace(instances[train_index], split=TRAIN_SPLIT)
        return instances
