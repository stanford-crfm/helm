import os
from typing import List, Dict
import json

from helm.common.general import ensure_directory_exists, ensure_file_downloaded, flatten_list
from helm.common.hierarchical_logger import hlog
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

PID_TO_NAME = {
    "P136": "genre",
    "P1303": "instrument",
    "P50": "author",
    "P170": "creator",
    "P86": "composer",
    "P57": "director",
    "P1001": "applies_to_jurisdiction",
    "P4006": "overrules",
    "P3014": "laws_applied",
    "P2568": "repealed_by",
    "P5826": "majority_opinion_by",
    "P1620": "plaintiff",
    "P1591": "defendant",
    "P135": "movement",
    "P737": "influenced_by",
    "P102": "member_of_political_party",
    "P530": "diplomatic_relation",
    "P39": "position_held",
    "P122": "basic_form_of_government",
    "P1906": "office_held_by_head_of_state",
    "P35": "head_of_state",
    "P1313": "office_held_by_head_of_government",
    "P6": "head_of_government",
    "P47": "shares_border_with",
    "P30": "continent",
    "P36": "capital",
    "P1376": "capital_of",
    "P17": "country",
    "P190": "twinned_administrative_body",
    "P38": "currency",
    "P1304": "central_bank",
    "P355": "subsidiary",
    "P414": "stock_exchange",
    "P452": "industry",
    "P2384": "statement_describes",
    "P1136": "solved_by",
    "P277": "programming_language",
    "P1195": "file_extension",
    "P1141": "number_of_processor_cores",
    "P306": "operating_system",
    "P111": "measured_physical_quantity",
    "P8111": "recommended_unit_of_measurement",
    "P1086": "atomic_number",
    "P8000": "electron_configuration",
    "P61": "discoverer_or_inventor",
    "P575": "time_of_discovery_or_invention",
    "P189": "location_of_discovery",
    "P2175": "medical_condition_treated",
    "P2176": "drug_or_therapy_used_for_treatment",
    "P2293": "genetic_association",
    "P4044": "therapeutic_area",
    "P780": "symptoms_and_signs",
    "P19": "place_of_birth",
    "P20": "place_of_death",
    "P279": "subclass_of",
    "P37": "official_language",
    "P413": "position_played_on_team",
    "P54": "member_of_sports_team",
    "P166": "award_received",
    "P449": "original_network",
    "P69": "educated_at",
    "P138": "named_after",
    "P364": "original_language_of_film_or_TV_show",
    "P463": "member_of",
    "P101": "field_of_work",
    "P1923": "participating_team",
    "P106": "occupation",
    "P527": "has_part",
    "P176": "manufacturer",
    "P178": "developer",
    "P27": "country_of_citizenship",
    "P407": "language_of_work_or_name",
    "P131": "located_in_the_administrative_territorial_entity",
    "P1412": "languages_spoken_written_or_signed",
    "P108": "employer",
    "P264": "record_label",
    "P276": "location",
    "P937": "work_location",
    "P140": "religion",
    "P127": "owned_by",
    "P103": "native_language",
    "P31": "instance_of",
    "P495": "country_of_origin",
    "P159": "headquarters_location",
    "P740": "location_of_formation",
    "P361": "part_of",
}

NAME_TO_PID: Dict[str, str] = {v: k for k, v in PID_TO_NAME.items()}


class WIKIFactScenario(Scenario):
    """
    Fact Completion task using knowledge from WikiData.
    Data constructed using the dump at https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.gz

    We prompt models using the following format

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

    name = "wikifact"
    description = "Fact Completion in WikiData"
    tags = ["knowledge", "generation"]

    def __init__(self, subject: str):
        super().__init__()
        self.subject: str = subject
        assert subject in NAME_TO_PID, f"Invalid subject: {subject}"

    def get_instances(self, output_path: str) -> List[Instance]:
        # Download the raw data
        data_path: str = os.path.join(output_path, "data")
        ensure_directory_exists(data_path)

        # Read all the instances
        instances: List[Instance] = []
        splits = {
            "train": TRAIN_SPLIT,
            "dev": VALID_SPLIT,
            "test": TEST_SPLIT,
        }

        for split in splits:
            split_file_name: str = f"{split}.jsonl"
            json_path: str = os.path.join(data_path, split_file_name)
            ensure_file_downloaded(
                source_url="https://worksheets.codalab.org/rest/bundles/0x8c3b60eb7c6b462e822a150f194d3b35/"
                f"contents/blob/{split_file_name}",
                target_path=json_path,
                unpack=False,
            )

            hlog(f"Reading {json_path}")
            with open(json_path) as f:
                all_raw_data = f.readlines()
            for line in all_raw_data:
                raw_data = json.loads(line)
                if raw_data["property"] != NAME_TO_PID[self.subject]:
                    continue

                question: str = raw_data["template"]
                answers: List[str] = flatten_list(raw_data["result_names"])

                def answer_to_reference(answer: str) -> Reference:
                    return Reference(Output(text=answer.strip()), tags=[CORRECT_TAG])

                instance = Instance(
                    input=Input(text=question),
                    references=list(map(answer_to_reference, answers)),
                    split=splits[split],
                )
                instances.append(instance)

        return instances
