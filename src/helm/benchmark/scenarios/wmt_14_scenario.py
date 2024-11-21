from typing import List, Any
from datasets import load_dataset
from helm.common.hierarchical_logger import htrack_block
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TRAIN_SPLIT,
    VALID_SPLIT,
    TEST_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
)


MAX_TRAIN_INSTANCES = 20_000  # This is arbitrary, but 20,000 training examples should be enough.


class WMT14Scenario(Scenario):
    """
    The 2014 Workshop on Statistical Machine Translation:
    https://aclanthology.org/W14-3302.pdf

    The scenario consists of 5 subsets, each of which is a parallel corpus between English and another language. The
    non-English languages include Czech, German, French, Hindi, and Russian.

    For each language pair, the validation and test set each includes around 3,000 examples, while the training set is
    usually much larger. We therefore randomly downsample the training set to speedup data processing.

    Task prompt structure:

        Translate {source_language} to {target_language}:
        {Hypothesis} = {Reference}

    Example from WMT14 Fr-En:

        Hypothesis: Assemblée générale
        Reference: General Assembly
    """

    name = "WMT_14"
    description = "Scenario for the 2014 Workshop on Statistical Machine Translation"
    tags = ["machine_translation"]

    def __init__(self, source_language, target_language):
        super().__init__()
        valid_languages = set(["cs", "de", "fr", "hi", "ru", "en"])
        self.source_language = source_language
        self.target_language = target_language
        if self.source_language not in valid_languages or self.target_language not in valid_languages:
            raise ValueError("WMT14 only includes the following languages: cs, de, fr, hi, ru, en.")
        if self.source_language == self.target_language:
            raise ValueError("The source language and the target language should be different.")
        if self.source_language != "en" and self.target_language != "en":
            raise ValueError("One of the languages should be English.")

    def _deduplicate(self, dataset: List):
        """
        Remove instances in the dataset with the same label.
        """

        deduplicated_dataset = []
        seen_labels = set()
        for example in dataset:
            if example[self.target_language] not in seen_labels:
                seen_labels.add(example[self.target_language])
                deduplicated_dataset.append(example)
        return deduplicated_dataset

    def get_instances(self, output_path: str) -> List[Instance]:
        with htrack_block("Loading the HuggingFace dataset. The first time could take several minutes."):
            subset_name = f"{self.source_language if self.source_language!='en' else self.target_language}-en"
            hf_dataset: Any = load_dataset(
                "wmt14", subset_name, trust_remote_code=True, revision="b199e406369ec1b7634206d3ded5ba45de2fe696"
            )
            splits = {"train": TRAIN_SPLIT, "validation": VALID_SPLIT, "test": TEST_SPLIT}

        instances: List[Instance] = []
        with htrack_block("Generating instances"):
            # Some training sets are too large, so we will only take a random subset of it.
            hf_dataset["train"] = hf_dataset["train"].shuffle(seed=42)[:MAX_TRAIN_INSTANCES]
            hf_dataset["train"]["translation"] = self._deduplicate(hf_dataset["train"]["translation"])
            for example in hf_dataset["train"]["translation"]:
                source_sentence: str = example[self.source_language]
                target_sentence: str = example[self.target_language]
                instances.append(
                    Instance(
                        input=Input(text=source_sentence),
                        references=[Reference(Output(text=target_sentence), tags=[CORRECT_TAG])],
                        split="train",
                    )
                )

        # No special handling needed for validation or test.
        for split_name in ["validation", "test"]:
            split = splits[split_name]
            for example in hf_dataset[split_name]:
                source_sentence = example["translation"][self.source_language]
                target_sentence = example["translation"][self.target_language]
                instances.append(
                    Instance(
                        input=Input(text=source_sentence),
                        references=[Reference(Output(text=target_sentence), tags=[CORRECT_TAG])],
                        split=split,
                    )
                )
        return instances
