import os
from typing import List, Dict
import json
import re

from helm.common.general import ensure_file_downloaded, ensure_directory_exists
from helm.benchmark.scenarios.scenario import Scenario, Instance, Reference, TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT, CORRECT_TAG, Input, Output

TAG_DICT = {
    "kpi": "Key Performance Indicators expressible in numerical and monetary value",
    "cy": "Current Year monetary value",
    "py": "Prior Year monetary value",
    "py1": "Two Year Past Value",
}
TAG_PAREN_RE = (r"\[", r"\]")
TAG_PAREN = tuple((e.strip("\\") for e in TAG_PAREN_RE))
TAG_PAREN_ESC = ("(", ")")
DATASET_SPLIT_TO_HELM_SPLIT = {"train": TRAIN_SPLIT, "valid": VALID_SPLIT, "test": TEST_SPLIT}


class KPIEDGARScenario(Scenario):
    """
        Paper:
        T. Deußer et al.,
        “KPI-EDGAR: A Novel Dataset and Accompanying Metric for Relation Extraction from Financial Documents.” 2022.
        https://arxiv.org/abs/2210.09163

        Website:
        https://github.com/tobideusser/kpi-edgar

        This is a dataset for Named Entity Recognition task for financial domain.

        Concretely, we prompt models using the following format:

        ```
    Context: {Sentence}
    Task: Extract key performance indicators (KPIs) and values from the above text. Also, specify one of the following categories to each of the extracted KPIs and values in brackets.
    kpi: Key Performance Indicators expressible in numerical and monetary value, cy: Current Year monetary value, py: Prior Year monetary value, py1: Two Year Past Value.
    Answer:
        ```

        Example

        ```
    Context: The following table summarizes our total share-based compensation expense and excess tax benefits recognized : As of December 28 , 2019 , there was $ 284 million of total unrecognized compensation cost related to nonvested share-based compensation grants .
    Task: Extract key performance indicators (KPIs) and values from the above text. Also, specify one of the following categories to each of the extracted KPIs and values in brackets.
    kpi: Key Performance Indicators expressible in numerical and monetary value, cy: Current Year monetary value, py: Prior Year monetary value, py1: Two Year Past Value.
    Answer:
        ```

        Reference:
        ```
        284 [cy], total unrecognized compensation cost [kpi]
        ```

    """  # noqa: E501

    name = "kpi_edgar"
    description = "Named Entity Recognition from financial documents."
    tags = ["named_entity_recognition", "finance"]

    JSON_URL = "https://raw.githubusercontent.com/tobideusser/kpi-edgar/2ec7084dcd55b4979bbe288d4aa1e962c685c9ab/data/kpi_edgar.json"
    JSON_FILENAME = "kpi_edgar.json"

    @staticmethod
    def get_sentences(dataset_obj: List[Dict]) -> List[Dict]:
        sentences = []
        for doc in dataset_obj:
            segments = doc["segments"]
            assert isinstance(segments, list)
            for segment in segments:
                segment_sentences = segment.get("sentences")
                if isinstance(segment_sentences, list):
                    for segment_sentence in segment_sentences:
                        sentences.append(segment_sentence)
        return sentences

    @staticmethod
    def escape_parenthesis(text: str) -> str:
        tmp0 = re.sub(TAG_PAREN_RE[0], TAG_PAREN_ESC[0], text)
        tmp1 = re.sub(TAG_PAREN_RE[1], TAG_PAREN_ESC[1], tmp0)
        return tmp1

    @staticmethod
    def get_output_text(
        words: List[str],
        annotations: List[Dict],
    ) -> str:
        def get_entity_for_annotation(words: List[str], annotation: Dict):
            start_idx = annotation["start"]
            end_idx = annotation["end"]
            annotated_words = words[start_idx:end_idx]
            phrase = KPIEDGARScenario.escape_parenthesis(" ".join(annotated_words))
            return "%s %s%s%s" % (phrase, TAG_PAREN[0], annotation["type_"], TAG_PAREN[1])

        entities = [get_entity_for_annotation(words, anno) for anno in annotations]
        return ", ".join(entities)

    @staticmethod
    def sentences_to_instances(sentences: List[Dict]) -> List[Instance]:
        tag_descriptions = ", ".join(["%s: %s" % (key, val) for (key, val) in TAG_DICT.items()]) + "."
        instances = []
        for sentence in sentences:
            words = [word_dict["value"] for word_dict in sentence["words"]]
            annotations = sentence["entities_anno"]
            passage = KPIEDGARScenario.escape_parenthesis(" ".join(words))
            input_text = (
                "Context: %s\n"
                "Task: Extract key performance indicators (KPIs) and values from the above text. Also, specify one of the following categories to each of the extracted KPIs and values in brackets.\n"
                "%s" % (passage, tag_descriptions)
            )
            dataset_split = sentence["split_type"]
            if dataset_split is None:
                continue
            split = DATASET_SPLIT_TO_HELM_SPLIT[dataset_split]
            output_text = KPIEDGARScenario.get_output_text(words, annotations)
            instances.append(Instance(
                input=Input(text=input_text),
                references=[Reference(Output(text=output_text), tags=[CORRECT_TAG])],
                split=split,
            ))
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
