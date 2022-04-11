import os
from typing import List
import json

from common.general import ensure_file_downloaded, flatten_list
from common.hierarchical_logger import hlog
from .scenario import Scenario, Instance, Reference, TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT, CORRECT_TAG


class WIKIScenario(Scenario):
    """
    Fact Completion task using knowledge from WikiData.
    Data constructed using the dump at https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.gz

    We prompt models using the following format:
        Input sequence:
            <subject> <predicate>
        Output Sequence (Target completion):
            <object>
    Using an example from the training dataset, we have
        Doug Eckerty is an instance of human
        Chegerd, Khash is an instance of village
        S. George Ellsworth is an instance of
        Target completion:
            human
    """

    name = "wiki"
    description = "Fact Completion in WikiData"
    tags = ["knowledge", "generation"]

    def __init__(self, subject: str):
        self.subject = subject
        assert subject in [
            "P31",
            "P17",
            "P131",
            "P279",
            "P20",
            "P19",
            "P27",
            "P106",
            "P127",
            "P138",
            "P30",
            "P361",
            "P407",
            "P50",
            "P136",
            "P527",
            "P57",
            "P364",
            "P495",
            "P69",
            "P1412",
            "P47",
            "P413",
            "P54",
            "P159",
            "P170",
            "P276",
            "P108",
            "P176",
            "P86",
            "P264",
            "P1923",
            "P1303",
            "P449",
            "P102",
            "P101",
            "P140",
            "P39",
            "P452",
            "P463",
            "P103",
            "P166",
            "P1001",
            "P61",
            "P178",
            "P36",
            "P306",
            "P277",
            "P937",
            "P6",
            "P737",
            "P1906",
            "P5826",
            "P740",
            "P135",
            "P414",
            "P1313",
            "P1591",
            "P1376",
            "P2176",
            "P355",
            "P37",
            "P189",
            "P2175",
            "P2568",
            "P122",
            "P190",
            "P2293",
            "P38",
            "P780",
            "P111",
            "P35",
            "P2384",
            "P8111",
            "P530",
            "P1304",
            "P3014",
            "P1620",
            "P1136",
            "P4006",
            "P4044",
        ]

    def get_instances(self) -> List[Instance]:
        # Download the raw data
        data_path = os.path.join(self.output_path, "data")
        ensure_file_downloaded(
            source_url="https://snap.stanford.edu/betae/fact_completion_data.zip",
            target_path=data_path,
            unpack=True,
            unpack_type="unzip",
        )
        # Read all the instances
        instances: List[Instance] = []
        splits = {
            "train": TRAIN_SPLIT,
            "dev": VALID_SPLIT,
            "test": TEST_SPLIT,
        }

        for split in splits:
            json_path = os.path.join(data_path, f"{split}.jsonl")

            hlog(f"Reading {json_path}")
            with open(json_path) as f:
                all_raw_data = f.readlines()
            for line in all_raw_data:
                raw_data = json.loads(line)
                if raw_data["property"] != self.subject:
                    continue
                question = raw_data["template"] + ' '
                answers = flatten_list(raw_data["result_names"])

                def answer_to_reference(answer):
                    return Reference(output=answer.strip(), tags=[CORRECT_TAG])

                instance = Instance(
                    input=question, references=list(map(answer_to_reference, answers)), split=splits[split],
                )
                instances.append(instance)

        return instances
