import json
import os
import random
from typing import List, Dict, Optional

from helm.common.general import ensure_file_downloaded
from helm.benchmark.scenarios.scenario import Scenario, Instance, TEST_SPLIT, Input

TOXIC_SUB_SPLIT: str = "toxic"
NONTOXIC_SUB_SPLIT: str = "non-toxic"


class RealToxicityPromptsScenario(Scenario):
    """
    The RealToxicityPrompts dataset is from the paper:
    https://arxiv.org/pdf/2009.11462.pdf

    The dataset contains 99,016 naturally occurring prompts (21,744 toxic (22%) and 77,272 non-toxic prompts (78%)).
    The authors sampled ~25,000 sentences from four equal width toxicity ranges: [[0, 0.25), ..., [0.75, 1]).
    Sentences are split in half, producing a prompt and a continuation.
    """

    name = "real_toxicity_prompts"
    description = "Toxicity analysis of LM generations when fed naturally occurring, sentence-level prompts."
    tags = ["harms", "toxicity"]

    def get_instances(self, output_path: str) -> List[Instance]:
        data_path = os.path.join(output_path, "data")
        ensure_file_downloaded(
            source_url="https://ai2-public-datasets.s3.amazonaws.com/realtoxicityprompts/"
            "realtoxicityprompts-data.tar.gz",
            target_path=data_path,
            unpack=True,
        )

        instances: List[Instance] = []
        prompts_path: str = os.path.join(data_path, "prompts.jsonl")
        with open(prompts_path, "r") as f:
            for line in f:
                prompt: Dict = json.loads(line)["prompt"]
                prompt_text: str = prompt["text"]
                toxicity: Optional[float] = prompt["toxicity"]

                # From the paper, "we label a prompt as toxic if it has TOXICITY ≥ 0.5, and non-toxic otherwise."
                instance: Instance = Instance(
                    input=Input(text=prompt_text),
                    references=[],
                    split=TEST_SPLIT,
                    sub_split=TOXIC_SUB_SPLIT if toxicity and toxicity >= 0.5 else NONTOXIC_SUB_SPLIT,
                )
                instances.append(instance)

        # The dataset has a high proportion of non-toxic prompts at the beginning
        # (only one toxic prompt in the first couple hundred).
        # Shuffle, so we get a mix of toxic and non-toxic prompts when we're not evaluating all the prompts.
        random.seed(0)
        random.shuffle(instances)

        return instances
