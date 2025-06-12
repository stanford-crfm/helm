from typing import List, Optional
import json
import os
import re

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
from helm.common.general import ensure_file_downloaded, ensure_directory_exists


class EchrJudgeScenario(Scenario):
    """The "Binary Violation" Classification task from the paper Neural Legal Judgment Prediction in English [(Chalkidis et al., 2019)](https://arxiv.org/pdf/1906.02059.pdf).

    The task is to analyze the description of a legal case from the European Court of Human Rights (ECHR),
    and classify it as positive if any human rights article or protocol has been violated and negative otherwise.

    The case text can be very long, which sometimes results in incorrect model output
    when using zero-shot predictions in many cases.
    Therefore, have added two trivial cases to the instructions part.

    Example Prompt:
        Is the following case a violation of human rights?  (Instructions)

        Case: Human rights have not been violated.          (Trivial No case in instructions)
        Answer: No

        Case: Human rights have been violated.              (Trivial Yes case in instructions)
        Answer: Yes

        Case: <TEXT>                                        (In-context examples, if possible)
        Answer: <Label>                                     (Label is correct answer, Yes or No)

        ...
        Case: <TEXT>                                        (Target input text)
        Answer: <Output>                                    (Output ::= Yes | No)
    """  # noqa: E501

    # Names of the tasks we support
    name = "echr_judgment_classification"
    description = 'The "Binary Violation" Classification task from the paper Neural Legal Judgment Prediction in English [(Chalkidis et al., 2019)](https://arxiv.org/pdf/1906.02059.pdf).'  # noqa: E501
    tags = ["classification", "judgement", "legal"]

    # Dataset file name
    _DATASET_URL = "https://archive.org/download/ECHR-ACL2019/ECHR_Dataset.zip"

    # Answer labels
    YES_ANSWER = "Yes"
    NO_ANSWER = "No"

    # Prompt constants (used in adapter)
    PROMPT_INPUT = "Case"
    PROMPT_OUTPUT = "Answer"

    YES_EX = f"\n\n{PROMPT_INPUT}: Human rights have been violated.\n{PROMPT_OUTPUT}: {YES_ANSWER}"
    NO_EX = f"\n\n{PROMPT_INPUT}: Human rights have not been violated.\n{PROMPT_OUTPUT}: {NO_ANSWER}"
    INST_EX = f"{NO_EX}{YES_EX}"

    PROMPT_INST = "Is the following case a violation of human rights?"  # Prompt for instructions
    PROMPT_INST_WITH_EX = f"{PROMPT_INST}{INST_EX}"  # Prompt for instructions with trivial examples

    def __init__(self, filter_max_length: Optional[int] = None):
        """
        Args:
            filter_max_length: Int indicating maximum length for documents. Documents longer than
                               train_filter_max_length tokens (using whitespace tokenization)
                               will be filtered out.
        """
        super().__init__()
        self.filter_max_length = filter_max_length

    def count_words(self, text: str) -> int:
        """Returns the number of words in the text"""
        return len(re.split(r"\W+", text))

    def get_instances(self, output_path: str) -> List[Instance]:
        data_dir = os.path.join(output_path, "data")
        ensure_directory_exists(data_dir)
        file_name = self._DATASET_URL.split("/")[-1]
        download_directory_path = os.path.join(data_dir, self._DATASET_URL)
        ensure_file_downloaded(
            source_url=self._DATASET_URL,
            target_path=download_directory_path,
            unpack=True,
            unpack_type="unzip",
        )

        source_split_to_helm_split = {"EN_train": TRAIN_SPLIT, "EN_dev": VALID_SPLIT, "EN_test": TEST_SPLIT}

        instances: List[Instance] = []
        for source_split, helm_split in source_split_to_helm_split.items():
            target_data_dir = os.path.join(download_directory_path, source_split)
            for file_name in os.listdir(target_data_dir):
                if not file_name.endswith(".json"):
                    continue
                file_path = os.path.join(target_data_dir, file_name)
                with open(file_path, "r") as f:
                    raw_data = json.load(f)
                input_text = " ".join(raw_data["TEXT"])
                if self.filter_max_length is not None and self.count_words(input_text) > self.filter_max_length:
                    continue
                answer = self.YES_ANSWER if len(raw_data["VIOLATED_ARTICLES"]) > 0 else self.NO_ANSWER
                correct_reference = Reference(Output(text=answer), tags=[CORRECT_TAG])
                instances.append(Instance(input=Input(input_text), references=[correct_reference], split=helm_split))
        return instances
