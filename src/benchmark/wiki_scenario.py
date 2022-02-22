import os
from typing import List
import json

from common.general import ensure_file_downloaded
from common.hierarchical_logger import hlog
from .scenario import Scenario, Instance, Reference, TRAIN_TAG, VALID_TAG, TEST_TAG, CORRECT_TAG


def flatten_list(l: list):
    return sum(map(flatten_list, l), []) if isinstance(l, list) else [l]


class WIKIScenario(Scenario):
    """
    Fact Completion task using knowledge from WikiData.
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
        instances = []
        splits = {
            "train": TRAIN_TAG,
            "dev": VALID_TAG,
            "test": TEST_TAG,
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
                question, answers = raw_data["template"], flatten_list(raw_data["result_names"])

                def answer_to_reference(answer):
                    return Reference(output=" " + answer.strip(), tags=[CORRECT_TAG])

                instance = Instance(
                    input=question, references=list(map(answer_to_reference, answers)), tags=[splits[split]],
                )
                instances.append(instance)

        return instances
