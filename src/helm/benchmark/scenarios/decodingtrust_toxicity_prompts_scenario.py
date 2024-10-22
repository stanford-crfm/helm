import json
import os
import random
from typing import List, Dict

from helm.common.general import ensure_file_downloaded
from helm.benchmark.scenarios.scenario import Scenario, Instance, TEST_SPLIT, Input


DATA_REPO_HASH = "38972f6ccbf376a8d0660babafb4d2b3b9cca3f4"


class DecodingTrustToxicityPromptsScenario(Scenario):
    """
    The DecodingTrustToxicityPrompts dataset is from the paper:
    https://arxiv.org/abs//2306.11698

    The dataset contains 99,016 naturally occurring prompts (21,744 toxic (22%) and 77,272 non-toxic prompts (78%)).
    The authors sampled ~25,000 sentences from four equal width toxicity ranges: [[0, 0.25), ..., [0.75, 1]).
    Sentences are split in half, producing a prompt and a continuation.
    """

    name = "decodingtrust_toxicity_prompts"
    description = "Toxicity analysis of LM generations when fed naturally occurring, sentence-level prompts."
    tags = ["harms", "toxicity"]

    def __init__(self, subject: str):
        super().__init__()
        self.subject: str = subject

    def get_instances(self, output_path: str) -> List[Instance]:
        source_url = (
            f"https://raw.githubusercontent.com/AI-secure/DecodingTrust-Data-Legacy/{DATA_REPO_HASH}/data/toxicity/"
            f"user_prompts/{{}}.jsonl"
        )

        TOXIC_SUB_SPLIT: str = "toxic"
        NONTOXIC_SUB_SPLIT: str = "non-toxic"
        GPT3_SUB_SPLIT: str = "gpt-3.5"
        GPT4_SUB_SPLIT: str = "gpt-4"
        FILENAMES = {
            TOXIC_SUB_SPLIT: "toxic",
            NONTOXIC_SUB_SPLIT: "nontoxic",
            GPT3_SUB_SPLIT: "toxic.jsonl.gpt3.n=25.out1.split.annotated.jsonl.filtered",
            GPT4_SUB_SPLIT: "toxic.jsonl.gpt4.n=25.out1.split.annotated.jsonl.filtered",
        }

        source_url = source_url.format(FILENAMES[self.subject])
        data_path = os.path.join(output_path, FILENAMES[self.subject])

        ensure_file_downloaded(
            source_url=source_url,  # to be filled
            target_path=data_path,
        )

        instances: List[Instance] = []

        prompts_path: str = data_path
        with open(prompts_path, "r") as f:
            for line in f:
                prompt: Dict = json.loads(line)["prompt"]
                prompt_text: str = prompt["text"]

                # From the paper, "we label a prompt as toxic if it has TOXICITY ≥ 0.5, and non-toxic otherwise."
                instance: Instance = Instance(
                    input=Input(text=prompt_text),
                    references=[],
                    split=TEST_SPLIT,
                )
                instances.append(instance)

        # The dataset has a high proportion of non-toxic prompts at the beginning
        # (only one toxic prompt in the first couple hundred).
        # Shuffle, so we get a mix of toxic and non-toxic prompts when we're not evaluating all the prompts.
        random.seed(0)
        random.shuffle(instances)

        return instances
