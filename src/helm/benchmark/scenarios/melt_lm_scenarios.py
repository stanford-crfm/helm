from typing import Any, Dict, List, Optional

import random
from datasets import load_dataset
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


class MELTLMMaskFillingScenario(Scenario):
    """
    Scenario for the MELT Masked Language Modeling dataset.
    """

    name = "melt_lm_mask_filling"
    description = "Masked Language Modeling scenario."
    tags = ["language_modeling", "mask_filling"]

    def __init__(
        self,
        dataset_name: str,
        revision: str,
        masked_ratio: float = 0.1,
        text_key: str = "text",
        subset: Optional[str] = None,
        splits: Optional[Dict[str, str]] = None,
    ):
        """Initializes the question answering scenario.

        Args:
            dataset_name: The name of the dataset.
            revision: The revision of the dataset to use.
            masked_ratio: The ratio of tokens to mask in the input text. Defaults to 0.1.
            text_key: The key to use for the text in the dataset. Defaults to "text".
            subset: The subset of the dataset to use. Defaults to "".
            splits: The splits to use for the dataset. Defaults to None.
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.subset = subset
        self.masked_ratio = masked_ratio
        self.text_key = text_key
        self.revision = revision
        self.splits = splits

    def _mask_text(self, text: str) -> str:
        """
        Mask a portion of the input text.
        Args:
            text (str): The input text to mask.
        Returns:
            str: The masked text.
        """
        tokens = text.split(" ")
        num_tokens_to_mask = int(len(tokens) * self.masked_ratio)
        indices_to_mask = random.sample(range(len(tokens)), num_tokens_to_mask)
        for index in indices_to_mask:
            tokens[index] = "[MASK]"
        return " ".join(tokens)

    def get_instances_for_splits(self, splits: Dict[str, str]) -> List[Instance]:
        """
        Helper for generating instances for a split.
        Args:
            splits (dict): Which splits to partition the data into.
        Returns:
            List[Instance]: Instances from the file for the specified split.
        """
        instances: List[Instance] = []
        dataset: Any = load_dataset(
            self.dataset_name,
            self.subset,
            revision=self.revision,
            trust_remote_code=True,
        )

        for dataset_split_name, helm_split_name in splits.items():
            for sample in dataset[dataset_split_name]:
                target_sentence = sample[self.text_key]
                source_sentence = self._mask_text(target_sentence)
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

        random.seed(42)
        instances: List[Instance] = self.get_instances_for_splits(splits=splits)
        return instances


class MELTLMMaskFillingMLQAScenario(MELTLMMaskFillingScenario):
    """
    Scenario for the MLQA dataset.
    This dataset is a multilingual question answering dataset.
    It contains questions in multiple languages and their corresponding
    answers in the same language. In this scenario, we are using the
    context of questions in the Vietnamese subset of the MLQA dataset.
    """

    name = "melt_lm_mask_filling_mlqa"
    description = "MLQA dataset for masked language modeling."
    tags = ["language_modeling", "mask_filling"]

    def __init__(self):
        super().__init__(
            dataset_name="facebook/mlqa",
            revision="397ed406c1a7902140303e7faf60fff35b58d285",
            subset="mlqa.vi.vi",
            text_key="context",
            splits={
                VALID_SPLIT: "validation",
                TEST_SPLIT: "test",
            },
        )


class MELTLMSpellingCorrectionScenario(Scenario):
    """
    Scenario for the MELT spelling correction dataset.
    """

    name = "melt_lm_spelling_correction"
    description = "Spelling Correction scenario."
    tags = ["language_modeling", "spelling_correction"]

    def __init__(
        self,
        dataset_name: str,
        revision: str,
        source_key: str = "text",
        target_key: str = "corrected_text",
        subset: Optional[str] = None,
        splits: Optional[Dict[str, str]] = None,
    ):
        """Initializes the question answering scenario.

        Args:
            dataset_name: The name of the dataset.
            revision: The revision of the dataset to use.
            source_key: The key to use for the source text in the dataset. Defaults to "text".
            target_key: The key to use for the target text in the dataset. Defaults to "corrected_text".
            subset: The subset of the dataset to use. Defaults to "".
            splits: The splits to use for the dataset. Defaults to None.
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.subset = subset
        self.source_key = source_key
        self.target_key = target_key
        self.revision = revision
        self.splits = splits

    def get_instances_for_splits(self, splits: Dict[str, str]) -> List[Instance]:
        """
        Helper for generating instances for a split.
        Args:
            splits (dict): Which splits to partition the data into.
        Returns:
            List[Instance]: Instances from the file for the specified split.
        """
        instances: List[Instance] = []
        dataset: Any = load_dataset(
            self.dataset_name,
            self.subset,
            revision=self.revision,
            trust_remote_code=True,
        )
        if len(splits) == 1:
            all_keys = list(splits.keys())
            dataset = dataset[all_keys[0]].train_test_split(test_size=0.33, seed=42)
            splits = {
                "train": TRAIN_SPLIT,
                "test": TEST_SPLIT,
            }

        for dataset_split_name, helm_split_name in splits.items():
            for sample in dataset[dataset_split_name]:
                source_sentence = sample[self.source_key]
                target_sentence = sample[self.target_key]
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

        random.seed(42)
        instances: List[Instance] = self.get_instances_for_splits(splits=splits)
        return instances


class MELTLMSpellingCorrectionVSECScenario(MELTLMSpellingCorrectionScenario):
    """
    Scenario for the VSEC dataset.
    The VSEC dataset is a Vietnamese spelling correction dataset.
    It contains 9,341 pairs of sentences where the first sentence is a misspelled
    version of the second sentence, which is the correct version.
    The mistakes are common spelling errors made by Vietnamese speakers and typists.
    """

    name = "melt_lm_spelling_correction_vsec"
    description = "VSEC dataset for spelling correction."
    tags = ["language_modeling", "spelling_correction"]

    def __init__(self):
        super().__init__(
            dataset_name="ura-hcmut/VSEC",
            revision="a6732e131605b5ec24ecc1745c6061c5ae86814e",
            source_key="text",
            target_key="correct",
            splits={
                TEST_SPLIT: "test",
            },
        )
