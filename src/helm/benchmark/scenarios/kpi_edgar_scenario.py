import os
from typing import List, Dict
import json
import re

from helm.common.general import ensure_file_downloaded, ensure_directory_exists
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


class KPIEDGARScenario(Scenario):
    """A financial named entity recognition (NER) scenario based on KPI-EDGAR (T. Deußer et al., 2022).

    This scenario has been modified from the paper. The original paper has 12 entity types and requires the model
    to extract pairs of related entities. This scenario only use four named entity types (kpi, cy, py, py1) and only
    requires the model to extract individual entities.

    Paper:
    T. Deußer et al.,
    “KPI-EDGAR: A Novel Dataset and Accompanying Metric for Relation Extraction from Financial Documents.” 2022.
    https://arxiv.org/abs/2210.09163

    Prompt format:

    ```
    Context: {Sentence}
    Task: Extract key performance indicators (KPIs) and values from the above text. Also, specify one of the following categories to each of the extracted KPIs and values in brackets.
    kpi: Key Performance Indicators expressible in numerical and monetary value, cy: Current Year monetary value, py: Prior Year monetary value, py1: Two Year Past Value.
    Answer:
    ```

    Example input:

    ```
    Context: The following table summarizes our total share-based compensation expense and excess tax benefits recognized : As of December 28 , 2019 , there was $ 284 million of total unrecognized compensation cost related to nonvested share-based compensation grants .
    Task: Extract key performance indicators (KPIs) and values from the above text. Also, specify one of the following categories to each of the extracted KPIs and values in brackets.
    kpi: Key Performance Indicators expressible in numerical and monetary value, cy: Current Year monetary value, py: Prior Year monetary value, py1: Two Year Past Value.
    Answer:
    ```

    Example reference:
    ```
    284 [cy], total unrecognized compensation cost [kpi]
    ```"""  # noqa: E501

    name = "kpi_edgar"
    description = "Named Entity Recognition from financial documents."
    tags = ["named_entity_recognition", "finance"]

    TAG_DICT = {
        "kpi": "Key Performance Indicators expressible in numerical and monetary value",
        "cy": "Current Year monetary value",
        "py": "Prior Year monetary value",
        "py1": "Two Year Past Value",
    }
    TAG_DESCRIPTIONS = ", ".join(["%s: %s" % (key, val) for (key, val) in TAG_DICT.items()]) + "."
    TAG_PAREN_RE = (r"\[", r"\]")
    TAG_PAREN = tuple((e.strip("\\") for e in TAG_PAREN_RE))
    TAG_PAREN_ESC = ("(", ")")
    DATASET_SPLIT_TO_HELM_SPLIT = {"train": TRAIN_SPLIT, "valid": VALID_SPLIT, "test": TEST_SPLIT}
    JSON_URL = "https://raw.githubusercontent.com/tobideusser/kpi-edgar/2ec7084dcd55b4979bbe288d4aa1e962c685c9ab/data/kpi_edgar.json"  # noqa: E501
    JSON_FILENAME = "kpi_edgar.json"

    @staticmethod
    def get_sentences(dataset: List[Dict]) -> List[Dict]:
        return [
            sentence
            for document in dataset
            for segment in document["segments"]
            for sentence in segment["sentences"] or []
        ]

    @staticmethod
    def escape_parenthesis(text: str) -> str:
        tmp0 = re.sub(KPIEDGARScenario.TAG_PAREN_RE[0], KPIEDGARScenario.TAG_PAREN_ESC[0], text)
        tmp1 = re.sub(KPIEDGARScenario.TAG_PAREN_RE[1], KPIEDGARScenario.TAG_PAREN_ESC[1], tmp0)
        return tmp1

    @staticmethod
    def get_output_text(
        words: List[str],
        annotations: List[Dict],
    ) -> str:
        # def get_entity_for_annotation(words: List[str], annotation: Dict) -> str
        entities: List[str] = []
        for annotation in annotations:
            annotation_type = annotation["type_"]
            if annotation_type not in KPIEDGARScenario.TAG_DICT:
                continue
            start_idx = annotation["start"]
            end_idx = annotation["end"]
            annotated_words = words[start_idx:end_idx]
            phrase = KPIEDGARScenario.escape_parenthesis(" ".join(annotated_words))
            entities.append(
                "%s %s%s%s" % (phrase, KPIEDGARScenario.TAG_PAREN[0], annotation_type, KPIEDGARScenario.TAG_PAREN[1])
            )

        return ", ".join(entities)

    @staticmethod
    def sentences_to_instances(sentences: List[Dict]) -> List[Instance]:
        instances: List[Instance] = []
        for sentence in sentences:
            dataset_split: str = sentence["split_type"]
            if dataset_split is None:
                continue
            split = KPIEDGARScenario.DATASET_SPLIT_TO_HELM_SPLIT[dataset_split]

            words: List[str] = [word_dict["value"] for word_dict in sentence["words"]]
            passage = KPIEDGARScenario.escape_parenthesis(" ".join(words))
            input_text = (
                "Context: %s\n"
                "Task: Extract key performance indicators (KPIs) and values from the above text. Also, specify one of the following categories to each of the extracted KPIs and values in brackets.\n"  # noqa: E501
                "%s" % (passage, KPIEDGARScenario.TAG_DESCRIPTIONS)
            )

            annotations = sentence["entities_anno"]
            output_text = KPIEDGARScenario.get_output_text(words, annotations)
            if not output_text:
                continue

            instances.append(
                Instance(
                    input=Input(text=input_text),
                    references=[Reference(Output(text=output_text), tags=[CORRECT_TAG])],
                    split=split,
                )
            )
        return instances

    def get_instances(self, output_path: str) -> List[Instance]:
        data_path = os.path.join(output_path, "data")
        ensure_directory_exists(data_path)

        base_url = self.JSON_URL
        dataset_file_name = self.JSON_FILENAME
        target_path = os.path.join(data_path, dataset_file_name)
        ensure_file_downloaded(source_url=base_url, target_path=target_path)

        with open(target_path, "r") as f:
            raw_dataset = json.load(f)
        return KPIEDGARScenario.sentences_to_instances(KPIEDGARScenario.get_sentences(raw_dataset))
