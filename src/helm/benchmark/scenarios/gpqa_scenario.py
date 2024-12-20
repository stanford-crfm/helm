import datasets
import os
import random
from typing import List
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TRAIN_SPLIT,
    TEST_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
)
from helm.common.general import ensure_directory_exists


SUBSETS = ["gpqa_main", "gpqa_diamond", "gpqa_extended"]

# Train example indices below are found by indexing examples given in the original paper repo
TRAIN_EXAMPLE_INDICES = {
    "gpqa_main": [339, 105],
    "gpqa_diamond": [124, 39],
    "gpqa_extended": [146, 330, 436],
}


class GPQAScenario(Scenario):
    """GPQA

    GPQA is a multiple-choice, Q&A dataset of very hard questions written and validated by experts in biology, physics,
    and chemistry. When attempting questions out of their own domain (e.g., a physicist answers a chemistry question),
    these experts get only 34% accuracy, despite spending >30m with full access to Google."""

    name = "gpqa"
    description = "A Graduate-Level Google-Proof Q&A Benchmark"
    tags = ["question answering"]

    def __init__(self, subset: str, random_seed=42):
        super().__init__()
        assert subset in SUBSETS, "Unknown subset: {}".format(subset)
        self.subset = subset
        self.random_seed = random_seed

    def get_instances(self, output_path: str) -> List[Instance]:
        # Get GPQA from HuggingFace
        cache_dir = os.path.join(output_path, "data")
        ensure_directory_exists(cache_dir)
        dataset = datasets.load_dataset(
            "Idavidrein/gpqa",
            self.subset,
            trust_remote_code=True,
            cache_dir=cache_dir,
            split="train",
            revision="90b8e5be2b1d3d2dbfe016cdab47981150600c4a",
        )
        assert isinstance(dataset, datasets.Dataset)

        # Read all instances
        random.seed(self.random_seed)
        instances: List[Instance] = []
        for idx, row in enumerate(dataset):
            input = Input(text=row["Question"].strip())
            references = [
                Reference(Output(text=row["Correct Answer"].strip()), tags=[CORRECT_TAG]),
                Reference(Output(text=row["Incorrect Answer 1"].strip()), tags=[]),
                Reference(Output(text=row["Incorrect Answer 2"].strip()), tags=[]),
                Reference(Output(text=row["Incorrect Answer 3"].strip()), tags=[]),
            ]
            random.shuffle(references)
            if idx in TRAIN_EXAMPLE_INDICES[self.subset]:
                extra_data = {
                    "chain_of_thought": row["Explanation"],
                }
                instance = Instance(input=input, references=references, split=TRAIN_SPLIT, extra_data=extra_data)
            else:
                instance = Instance(input=input, references=references, split=TEST_SPLIT)
            instances.append(instance)

        return instances
