import os
from typing import List
from common.hierarchical_logger import hlog
from common.general import ensure_file_downloaded, ensure_directory_exists
from .scenario import Scenario, Instance, Reference, TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT, CORRECT_TAG
import json


class LegalSupportScenario(Scenario):
    """
    This dataset is the result of ongoing/yet-to-be-released work. For more questions
    on its construction, contact Neel Guha (nguha@stanford.edu). 

    The LegalSupport dataset tests a model's ability to determine which of two case parentheticals
    better support a specific legal assertion. Each sample contains: 

        - a short passage, ranging between 1-3 sentences. 
        - a first parenthetical, describing or quoting a particular case ("case_a")
        - a second parenthetical, describing or quoting a different case ("case_b")
        - a label ("a" or "b") corresponding to whether the first or second parenthetical better supports the 
        assertions made in the passage. 

    We prompt models using the following format:
        <passage>
        a. <case_a>
        b. <case_b>
        answer:

        Multiple choice:
            <answer>

    Using an example from the test dataset, we have
        passage: Rather, we hold the uniform rule is ... that of 'good moral character". Courts have also endorsed using federal, 
        instead of state, standards to interpret federal laws regulating immigration.
        
        a. Interpreting "adultery” for the purpose of eligibility for voluntary departure, 
        and holding that "the appropriate approach is the application of a uniform federal standard."
        
        b. Using state law to define "adultery” in the absence of a federal definition, and suggesting that 
        arguably, Congress intended to defer to the state in which an alien chooses to live for the precise 
        definition ... for it is that particular community which has the greatest interest in its residents moral character.

        Answer:
            a
    """

    name = "legal_support"
    description = "Binary multiple choice question dataset for legal argumentative reasoning."
    tags = ["question_answering", "law"]

    def __init__(self):
        pass

    def get_instances(self) -> List[Instance]:
        data_path = os.path.join(self.output_path, "data")
        ensure_file_downloaded(
            source_url="https://docs.google.com/uc?export=download&id=1PVoyddrCHChMxYrLhsI-zu7Xzs5S8N77",
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
                passage = raw_data["context"]
                answers = [raw_data["citation_a"]["parenthetical"], raw_data["citation_b"]["parenthetical"]]
                correct_choice = raw_data["label"]
                answers_dict = dict(zip(["a", "b"], answers))
                correct_answer = answers_dict[correct_choice]

                def answer_to_reference(answer):
                    return Reference(output=answer, tags=[CORRECT_TAG] if answer == correct_answer else [])

                instance = Instance(
                    input=passage, references=list(map(answer_to_reference, answers)), split=splits[split],
                )

                instances.append(instance)

        return instances
