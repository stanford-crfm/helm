import os
import random
from typing import List

import pandas as pd

from helm.benchmark.scenarios.scenario import (
    Input,
    Instance,
    Output,
    Reference,
    Scenario,
    CORRECT_TAG,
    TEST_SPLIT,
)
from helm.common.general import ensure_file_downloaded
from helm.common.hierarchical_logger import hlog

# BHASA Scenarios
#   D. Linguistic Diagnostics

# D. Linguistic Diagnostics (LINDSEA)
#   1. Syntax
#   2. Pragmatics


# 1. Syntax: LINDSEA Minimal Pairs
class LINDSEASyntaxMinimalPairsScenario(Scenario):
    """
    The LINDSEA Minimal Pairs dataset is a linguistic diagnostic scenario targeting syntactic phenomena.
    The data is manually handcrafted by linguists and native speakers and verified through multiple rounds
    of quality control. The high-level categories tested for include morphology, argument structure,
    filler-gap dependencies, as well as negative polarity items and negation.

    The test is designed as a minimal pair, with a pair of sentences that differ minimally from each other
    and which exemplify a specific syntactic phenomenon. The system under test needs to determine which
    sentence of the pair is more acceptable.

    The models are prompted using the following general format:

        Which sentence is more acceptable?
        Answer only with a single letter A or B.
        <sentence>

    Target completion:
        <sentence>

    @misc{leong2023bhasa,
        title={BHASA: A Holistic Southeast Asian Linguistic and Cultural Evaluation Suite for Large Language Models},
        author={Wei Qi Leong
            and Jian Gang Ngui
            and Yosephine Susanto
            and Hamsawardhini Rengarajan
            and Kengatharaiyer Sarveswaran
            and William Chandra Tjhi
        },
        year={2023},
        eprint={2309.06085},
        archivePrefix={arXiv},
        primaryClass={cs.CL},
        url={https://arxiv.org/abs/2309.06085},
    }
    """

    name = "lindsea_minimal_pairs"
    description = "LINDSEA minimal pairs task"
    tags = ["minimal_pairs", "linguistic_diagnostic", "syntax"]

    def __init__(self, method: str, language: str):
        super().__init__()
        self.method = method
        self.language = language
        self.prompts = {
            "id": {
                "instructions": "Kalimat mana yang lebih mungkin?",
                "output_prefix": "Jawablah dengan satu huruf saja, A atau B.",
            }
        }

    def download_dataset(self, output_path: str):
        BASE_URL = "https://raw.githubusercontent.com/aisingapore/BHASA/main/lindsea/"
        URLS = {
            "npis_and_negation": f"{BASE_URL}{self.language}/syntax/NPIs_and_negation.jsonl",
            "argument_structure": f"{BASE_URL}{self.language}/syntax/argument_structure.jsonl",
            "filler_gap_dependencies": f"{BASE_URL}{self.language}/syntax/filler-gap_dependencies.jsonl",
            "morphology": f"{BASE_URL}{self.language}/syntax/morphology.jsonl",
        }

        data_files = {}
        for file in list(URLS.keys()):
            target_path_file = os.path.join(output_path, file)
            ensure_file_downloaded(source_url=URLS[file], target_path=target_path_file)
            data_files[file] = pd.read_json(target_path_file, lines=True)
        dataset = pd.concat(data_files)

        return dataset

    def get_instances(self, output_path: str) -> List[Instance]:
        data = self.download_dataset(output_path)

        outputs = []
        if self.method == "mcq":
            category_list = data["category"].value_counts().keys()
            hlog("MCQ method for LINDSEA Minimal Pairs chosen. Shuffling options...")
            for category in category_list:
                # Fix shuffling within each category
                random.seed(1)
                for _, row in data[data["category"] == category].iterrows():
                    options = [(row["correct"], 1), (row["wrong"], 2)]
                    random.shuffle(options)
                    options_reversed = True if options[0][1] == 2 else False

                    prompt_components = self.prompts[self.language]
                    instructions = prompt_components["instructions"]
                    output_prefix = prompt_components["output_prefix"]
                    prompt = f"{instructions}\nA: {options[0][0]}\nB: {options[1][0]}\n{output_prefix}"
                    input = Input(text=prompt)
                    # Determine correct option based on whether shuffling reversed the options
                    references = [
                        Reference(Output(text="A"), tags=[] if options_reversed else [CORRECT_TAG]),
                        Reference(Output(text="B"), tags=[CORRECT_TAG] if options_reversed else []),
                    ]
                    instance = Instance(input=input, references=references, split=TEST_SPLIT)
                    outputs.append(instance)

        else:
            for _, row in data.iterrows():
                # No need to shuffle since we are comparing logprobs of the options separately
                input = Input(text="")
                references = [
                    Reference(Output(text=row["correct"].strip()), tags=[CORRECT_TAG]),
                    Reference(Output(text=row["wrong"].strip()), tags=[]),
                ]
                instance = Instance(
                    input=input,
                    references=references,
                    split=TEST_SPLIT,
                )
                outputs.append(instance)
        return outputs


# 2. Pragmatics
# 2.1 LINDSEA Pragmatic Reasoning (single sentence)
class LINDSEAPragmaticsPragmaticReasoningSingleScenario(Scenario):
    """
    The LINDSEA Pragmatic Reasoning dataset is a linguistic diagnostic scenario targeting pragmatics.
    The data is manually handcrafted by linguists and native speakers and verified through multiple rounds
    of quality control. The high-level categories tested for include scalar implicatures and presuppositions.

    The single-sentence pragmatic reasoning dataset involves questions targeting the truth value of a single sentence.
    The system under test needs to determine if the sentence is true/false or if the proposition is possible/impossible.

    The models are prompted using the following general format:

        Is the following statement true or false?
        Statement: <sentence>
        Answer only with True or False.

    Target completion:
        <answer>

    @misc{leong2023bhasa,
        title={BHASA: A Holistic Southeast Asian Linguistic and Cultural Evaluation Suite for Large Language Models},
        author={Wei Qi Leong
            and Jian Gang Ngui
            and Yosephine Susanto
            and Hamsawardhini Rengarajan
            and Kengatharaiyer Sarveswaran
            and William Chandra Tjhi
        },
        year={2023},
        eprint={2309.06085},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
    }
    """

    name = "lindsea_pragmatic_reasoning_single"
    description = "LINDSEA pragmatic reasoning single sentence task"
    tags = ["pragmatic_reasoning", "linguistic_diagnostic", "pragmatics"]

    def __init__(self, language: str):
        super().__init__()
        self.language = language
        self.prompt = {
            "id": {
                "question": "Apakah pernyataan berikut ini {}?",
                "instruction": "Jawablah dengan {} saja.",
            },
        }

    def download_dataset(self, output_path: str):
        BASE_URL = "https://raw.githubusercontent.com/aisingapore/BHASA/main/lindsea/"
        URL = f"{BASE_URL}{self.language}/pragmatics/pragmatic_reasoning_single.jsonl"
        file = "pragmatic_reasoning_single"
        target_path_file = os.path.join(output_path, file)
        ensure_file_downloaded(source_url=URL, target_path=target_path_file)
        dataset = pd.read_json(target_path_file, lines=True)
        return dataset

    def get_instances(self, output_path) -> List[Instance]:
        data = self.download_dataset(output_path)
        outputs = []
        for _, row in data.iterrows():
            passage = "{question}\nPernyataan: {text}\n{instruction}".format(
                question=self.prompt[self.language]["question"].format(row["question_translated"]),
                text=row["text"],
                instruction=self.prompt[self.language]["instruction"].format(row["choices_translated"]),
            )
            input = Input(text=passage)

            # Split "True or False" into ["True", "or", "False"]
            choices = row["choices"].split()
            choices_translated = row["choices_translated"].split()
            label2choice = {
                choices[0]: choices_translated[0],
                choices[2]: choices_translated[2],
            }
            references = [
                Reference(Output(text=label2choice[row["label"].strip()]), tags=[CORRECT_TAG]),
            ]
            instance = Instance(
                input=input,
                references=references,
                split=TEST_SPLIT,
            )
            outputs.append(instance)
        return outputs


# 2.2 Pragmatics: LINDSEA Pragmatic Reasoning (sentence pair)
class LINDSEAPragmaticsPragmaticReasoningPairScenario(Scenario):
    """
    The LINDSEA Pragmatic Reasoning dataset is a linguistic diagnostic scenario targeting pragmatics.
    The data is manually handcrafted by linguists and native speakers and verified through multiple rounds
    of quality control. The high-level categories tested for include scalar implicatures and presuppositions.

    The sentence-pair pragmatic reasoning dataset involves questions targeting whether a conclusion can be drawn
    from another sentence.

    The models are prompted using the following general format:

        Situation: <premise>
        Given this situation, is the following statement true or false?
        Statement: <hypothesis>
        Answer only with True or False.

    Target completion:
        <answer>

    @misc{leong2023bhasa,
        title={BHASA: A Holistic Southeast Asian Linguistic and Cultural Evaluation Suite for Large Language Models},
        author={Wei Qi Leong
            and Jian Gang Ngui
            and Yosephine Susanto
            and Hamsawardhini Rengarajan
            and Kengatharaiyer Sarveswaran
            and William Chandra Tjhi
        },
        year={2023},
        eprint={2309.06085},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
    }
    """

    name = "lindsea_pragmatic_reasoning_pair"
    description = "LINDSEA pragmatic reasoning sentence pair task"
    tags = ["pragmatic_reasoning", "linguistic_diagnostic", "pragmatics"]

    def __init__(self, language: str):
        super().__init__()
        self.language = language
        self.prompt = {
            "id": {
                "question": "Berdasarkan situasi ini, apakah pernyataan berikut ini benar atau salah?",
                "instruction": "Jawablah dengan Benar atau Salah saja.",
                True: "Benar",
                False: "Salah",
            },
        }

    def download_dataset(self, output_path: str):
        BASE_URL = "https://raw.githubusercontent.com/aisingapore/BHASA/main/lindsea/"
        URL = f"{BASE_URL}{self.language}/pragmatics/pragmatic_reasoning_pair.jsonl"
        file = "pragmatic_reasoning_pair"
        target_path_file = os.path.join(output_path, file)
        ensure_file_downloaded(source_url=URL, target_path=target_path_file)
        dataset = pd.read_json(target_path_file, lines=True)
        return dataset

    def get_instances(self, output_path) -> List[Instance]:
        data = self.download_dataset(output_path)
        outputs = []
        for _, row in data.iterrows():
            passage = "Situasi: {premise}\n{question}\nPernyataan: {conclusion}\n{instruction}".format(
                premise=row["text"],
                question=self.prompt[self.language]["question"],
                conclusion=row["conclusion"],
                instruction=self.prompt[self.language]["instruction"],
            )
            input = Input(text=passage)
            references = [
                Reference(Output(text=self.prompt[self.language][row["label"]]), tags=[CORRECT_TAG]),
            ]
            instance = Instance(
                input=input,
                references=references,
                split=TEST_SPLIT,
            )
            outputs.append(instance)
        return outputs
