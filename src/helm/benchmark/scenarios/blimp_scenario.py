import json
import os
from typing import List

from helm.common.general import ensure_file_downloaded
from helm.common.hierarchical_logger import hlog
from helm.benchmark.scenarios.scenario import CORRECT_TAG, Reference, Scenario, Instance, Input, TEST_SPLIT, Output


class BLiMPScenario(Scenario):
    """
    The BLiMP linguistic knowledge evaluation suite from this paper:
    https://aclanthology.org/2020.tacl-1.25.pdf

    BLiMP evaluates the linguistic knowledge of language models in terms of syntax, morphology,
    and semantics. The dataset covers 12 linguistic phenomena and 67 paradigms. There are 1000
    examples for each paradigm. Every example is a minimal pair that includes a good sentence
    and a bad sentence. The model is considered correct if it assigns a higher probability to
    the good sentence instead of the bad one.

    e.g. Good: Craig explored that grocery store. Bad: Craig explored that grocery stores.
    """

    name = "blimp"
    description = "The Benchmark of Linguistic Minimal Pairs for English"
    tags = ["linguistic_knowledge", "language_modeling", "minimal_pairs"]
    phenomenon_to_paradigms = {
        "island_effects": [
            "left_branch_island_simple_question",
            "sentential_subject_island",
            "wh_island",
            "coordinate_structure_constraint_object_extraction",
            "coordinate_structure_constraint_complex_left_branch",
            "adjunct_island",
            "complex_NP_island",
            "left_branch_island_echo_question",
        ],
        "anaphor_agreement": ["anaphor_gender_agreement", "anaphor_number_agreement"],
        "argument_structure": [
            "animate_subject_passive",
            "animate_subject_trans",
            "transitive",
            "passive_1",
            "intransitive",
            "passive_2",
            "causative",
            "drop_argument",
            "inchoative",
        ],
        "determiner_noun_agreement": [
            "determiner_noun_agreement_with_adj_irregular_2",
            "determiner_noun_agreement_1",
            "determiner_noun_agreement_with_adj_irregular_1",
            "determiner_noun_agreement_irregular_1",
            "determiner_noun_agreement_irregular_2",
            "determiner_noun_agreement_with_adjective_1",
            "determiner_noun_agreement_2",
            "determiner_noun_agreement_with_adj_2",
        ],
        "subject_verb_agreement": [
            "distractor_agreement_relative_clause",
            "irregular_plural_subject_verb_agreement_2",
            "irregular_plural_subject_verb_agreement_1",
            "regular_plural_subject_verb_agreement_1",
            "distractor_agreement_relational_noun",
            "regular_plural_subject_verb_agreement_2",
        ],
        "ellipsis": ["ellipsis_n_bar_1", "ellipsis_n_bar_2"],
        "control_raising": [
            "expletive_it_object_raising",
            "tough_vs_raising_1",
            "existential_there_subject_raising",
            "tough_vs_raising_2",
            "existential_there_object_raising",
        ],
        "quantifiers": [
            "existential_there_quantifiers_1",
            "superlative_quantifiers_1",
            "superlative_quantifiers_2",
            "existential_there_quantifiers_2",
        ],
        "irregular_forms": ["irregular_past_participle_verbs", "irregular_past_participle_adjectives"],
        "npi_licensing": [
            "sentential_negation_npi_licensor_present",
            "matrix_question_npi_licensor_present",
            "sentential_negation_npi_scope",
            "only_npi_licensor_present",
            "only_npi_scope",
            "npi_present_2",
            "npi_present_1",
        ],
        "binding": [
            "principle_A_domain_2",
            "principle_A_domain_3",
            "principle_A_case_1",
            "principle_A_case_2",
            "principle_A_reconstruction",
            "principle_A_c_command",
            "principle_A_domain_1",
        ],
        "filler_gap_dependency": [
            "wh_vs_that_with_gap",
            "wh_vs_that_no_gap",
            "wh_questions_object_gap",
            "wh_vs_that_no_gap_long_distance",
            "wh_questions_subject_gap",
            "wh_questions_subject_gap_long_distance",
            "wh_vs_that_with_gap_long_distance",
        ],
    }

    def __init__(self, phenomenon: str):
        super().__init__()
        assert phenomenon in self.phenomenon_to_paradigms, f"Unsupported phenomenon: {phenomenon}"
        self.phenomenon: str = phenomenon

    def get_instances(self, output_path: str) -> List[Instance]:
        # Download the raw data
        data_path: str = os.path.join(output_path, "data")
        ensure_file_downloaded(
            source_url="https://github.com/alexwarstadt/blimp/blob/master/BLiMP.zip?raw=true",
            target_path=data_path,
            unpack=True,
            unpack_type="unzip",
        )

        # Read all the instances
        instances: List[Instance] = []
        for paradigm in self.phenomenon_to_paradigms[self.phenomenon]:
            jsonl_path: str = os.path.join(data_path, "data", f"{paradigm}.jsonl")
            hlog(f"Reading {jsonl_path}")
            with open(jsonl_path) as f:
                for line in f:
                    # Example: {"sentence_good": "Who should Derek hug after shocking Richard?",
                    # "sentence_bad": "Who should Derek hug Richard after shocking?",
                    # "field": "syntax", "linguistics_term": "island_effects", "UID": "adjunct_island",
                    # "simple_LM_method": true, "one_prefix_method": false, "two_prefix_method": false,
                    # "lexically_identical": true, "pairID": "0"}
                    example = json.loads(line)
                    instance = Instance(
                        input=Input(text=""),
                        references=[
                            Reference(Output(text=example["sentence_good"]), tags=[CORRECT_TAG]),
                            Reference(Output(text=example["sentence_bad"]), tags=[]),
                        ],
                        split=TEST_SPLIT,
                    )
                    instances.append(instance)
        return instances
