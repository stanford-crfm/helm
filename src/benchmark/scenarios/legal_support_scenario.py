import json
import os
from typing import List

from common.hierarchical_logger import hlog
from common.general import ensure_file_downloaded
from .scenario import Scenario, Instance, Reference, TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT, CORRECT_TAG


class LegalSupportScenario(Scenario):
    """
    This dataset is the result of ongoing/yet-to-be-released work. For more questions
    on its construction, contact Neel Guha (nguha@stanford.edu).

    The LegalSupport dataset evaluates fine-grained reverse entailment. Each sample consists of a
    text passage making a legal claim, and two case summaries. Each summary describes a legal conclusion
    reached by a different court. The task is to determine which case (i.e. legal conclusion) most forcefully
    and directly supports the legal claim in the passage. The construction of this benchmark leverages
    annotations derived from a legal taxonomy expliciting different levels of entailment (e.g.
    "directly supports" vs "indirectly supports"). As such, the benchmark tests a model's ability to reason
    regarding the strength of support a particular case summary provides.

    The task is structured as multiple choice questions. There are two choices per question.

    Using an example from the test dataset, we have

    Input: Rather, we hold the uniform rule is ... that of 'good moral character". Courts have also endorsed
    using federal, instead of state, standards to interpret federal laws regulating immigration.

    Reference [CORRECT]: Interpreting "adultery” for the purpose of eligibility for voluntary departure,
    and holding that "the appropriate approach is the application of a uniform federal standard."

    Reference: Using state law to define "adultery” in the absence of a federal definition, and suggesting that
    arguably, Congress intended to defer to the state in which an alien chooses to live for the precise
    definition ... for it is that particular community which has the greatest interest in its residents moral
    character.
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
