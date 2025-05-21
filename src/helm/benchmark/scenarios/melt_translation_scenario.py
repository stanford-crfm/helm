from typing import Any, Dict, List, Optional

from datasets import load_dataset, Dataset
from helm.common.hierarchical_logger import htrack_block
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TRAIN_SPLIT,
    TEST_SPLIT,
    VALID_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
)


class MELTTranslationScenario(Scenario):
    name = "melt_translation"
    description = "Machine Translation scenario."
    tags = ["machine_translation"]

    def __init__(
        self,
        dataset_name: str,
        revision: str,
        source_language: str,
        target_language: str,
        subset: Optional[str] = None,
        splits: Optional[Dict[str, str]] = None,
    ):
        """Initializes the question answering scenario.

        Args:
            dataset_name: The name of the dataset.
            revision: The revision of the dataset to use.
            source_language: The source language to use.
            target_language: The target language to use.
            subset: The subset of the dataset to use. Defaults to "".
            splits: The splits to use for the dataset. Defaults to None.
        """
        super().__init__()
        self.MAX_TRAIN_INSTANCES = 20_000
        valid_languages = set(["vi", "en"])
        self.dataset_name = dataset_name
        self.subset = subset
        self.revision = revision
        self.splits = splits
        self.source_language = source_language
        self.target_language = target_language
        if self.source_language not in valid_languages or self.target_language not in valid_languages:
            raise ValueError("Supported languages: vi, en.")
        if self.source_language == self.target_language:
            raise ValueError("The source language and the target language should be different.")
        if self.source_language != "en" and self.target_language != "en":
            raise ValueError("One of the languages should be English.")

    def get_instances_for_splits(self, splits: Dict[str, str]) -> List[Instance]:
        """
        Helper for generating instances for a split.
        Args:
            splits (dict): Which splits to partition the data into.
        Returns:
            List[Instance]: Instances from the file for the specified split.
        """
        with htrack_block("Loading the HuggingFace dataset. The first time could take several minutes."):
            hf_dataset: Any = load_dataset(
                self.dataset_name,
                self.subset,
                revision=self.revision,
                trust_remote_code=True,
            )

        instances: List[Instance] = []

        for dataset_split_name, helm_split_name in splits.items():
            if helm_split_name == TRAIN_SPLIT:
                hf_dataset[dataset_split_name] = hf_dataset[dataset_split_name].shuffle(seed=42)[
                    : self.MAX_TRAIN_INSTANCES
                ]
                hf_dataset[dataset_split_name] = Dataset.from_dict(hf_dataset[dataset_split_name])

            for example in hf_dataset[dataset_split_name]:
                source_sentence = example[self.source_language]
                target_sentence = example[self.target_language]
                instances.append(
                    Instance(
                        input=Input(text=source_sentence),
                        references=[Reference(Output(text=target_sentence), tags=[CORRECT_TAG])],
                        split=helm_split_name,
                    )
                )
        return instances

    def get_instances(self, output_path: str) -> List[Instance]:
        if self.splits is None:
            splits = {"train": TRAIN_SPLIT, "validation": VALID_SPLIT, "test": TEST_SPLIT}
        else:
            splits = {}
            if "train" in self.splits:
                splits[self.splits[TRAIN_SPLIT]] = TRAIN_SPLIT
            if "validation" in self.splits:
                splits[self.splits[VALID_SPLIT]] = VALID_SPLIT
            if "test" in self.splits:
                splits[self.splits[TEST_SPLIT]] = TEST_SPLIT

        instances: List[Instance] = self.get_instances_for_splits(splits=splits)
        return instances


class MELTTranslationOPUS100Scenario(MELTTranslationScenario):
    """
    Scenario for the OPUS100 dataset.
    """

    name = "melt_translation_opus100"
    description = "OPUS100 dataset for machine translation."
    tags = ["machine_translation"]

    def __init__(self, **kwargs):
        super().__init__(
            dataset_name="vietgpt/opus100_envi",
            revision="45df06fb0b31edc882d7c8d34389261f995e5208",
            splits={
                TRAIN_SPLIT: "train",
                VALID_SPLIT: "validation",
                TEST_SPLIT: "test",
            },
            **kwargs,
        )


class MELTTranslationPhoMTScenario(MELTTranslationScenario):
    """
    Scenario for the PhoMT dataset.
    """

    name = "melt_translation_phomt"
    description = "PhoMT dataset for machine translation."
    tags = ["machine_translation"]

    def __init__(self, **kwargs):
        super().__init__(
            dataset_name="ura-hcmut/PhoMT",
            revision="74386685db01dc038860ff0a90d9f5fbde284bf7",
            splits={
                TRAIN_SPLIT: "train",
                VALID_SPLIT: "validation",
                TEST_SPLIT: "test",
            },
            **kwargs,
        )
