import json
import os
from typing import Dict, List

from helm.common.general import ensure_file_downloaded
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    CORRECT_TAG,
    TRAIN_SPLIT,
    VALID_SPLIT,
    Input,
    Output,
)


class MedMCQAScenario(Scenario):
    """
    From "MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering"
    (Pal et al.), MedMCQA is a "multiple-choice question answering (MCQA) dataset designed to address
    real-world medical entrance exam questions." The dataset "...has more than 194k high-quality AIIMS & NEET PG
    entrance exam MCQs covering 2.4k healthcare topics and 21 medical subjects are collected with an average
    token length of 12.77 and high topical diversity."

    The following is an example from the dataset:

    Question: In a patient of heart disease antibiotic prophylaxis for dental extraction is:
    A. Amoxicillin.
    B. Imipenem.
    C. Gentamicin.
    D. Erythromycin.
    Answer: A

    Paper: https://arxiv.org/abs/2203.14371
    Code: https://github.com/MedMCQA/MedMCQA

    @InProceedings{pmlr-v174-pal22a,
      title = 	  {MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering},
      author =    {Pal, Ankit and Umapathi, Logesh Kumar and Sankarasubbu, Malaikannan},
      booktitle = {Proceedings of the Conference on Health, Inference, and Learning},
      pages = 	 {248--260},
      year = 	 {2022},
      editor = 	 {Flores, Gerardo and Chen, George H and Pollard, Tom and Ho, Joyce C and Naumann, Tristan},
      volume = 	 {174},
      series = 	 {Proceedings of Machine Learning Research},
      month = 	 {07--08 Apr},
      publisher =    {PMLR},
      pdf = 	 {https://proceedings.mlr.press/v174/pal22a/pal22a.pdf},
      url = 	 {https://proceedings.mlr.press/v174/pal22a.html},
      abstract = {This paper introduces MedMCQA, a new large-scale, Multiple-Choice Question Answering (MCQA) dataset
      designed to address real-world medical entrance exam questions. More than 194k high-quality AIIMS & NEET PG
      entrance exam MCQs covering 2.4k healthcare topics and 21 medical subjects are collected with an average token
      length of 12.77 and high topical diversity. Each sample contains a question, correct answer(s), and other
      options which requires a deeper language understanding as it tests the 10+ reasoning abilities of a model across
      a wide range of medical subjects & topics. A detailed explanation of the solution, along with the above
      information, is provided in this study.}
    }
    """

    # From https://github.com/MedMCQA/MedMCQA#data-fields, there are four possible answer choices
    # where "cop" corresponds to the index of the correct option.
    ANSWER_OPTION_TO_INDEX: Dict[str, int] = {"opa": 1, "opb": 2, "opc": 3, "opd": 4}
    DATASET_DOWNLOAD_URL: str = (
        "https://drive.google.com/uc?export=download&id=15VkJdq5eyWIkfb_aoD3oS8i4tScbHYky&confirm=t"
    )

    name = "med_mcqa"
    description = (
        "MedMCQA is a multiple-choice question answering (MCQA) dataset designed to address "
        "real-world medical entrance exam questions."
    )
    tags = ["question_answering", "biomedical"]

    def get_instances(self, output_path: str) -> List[Instance]:
        data_path: str = os.path.join(output_path, "data")
        ensure_file_downloaded(
            source_url=self.DATASET_DOWNLOAD_URL,
            target_path=data_path,
            unpack=True,
            unpack_type="unzip",
        )

        instances: List[Instance] = []

        # From https://github.com/MedMCQA/MedMCQA#model-submission-and-test-set-evaluation,
        # "to preserve the integrity of test results, we do not release the test set's ground-truth to the public".
        for split in [TRAIN_SPLIT, VALID_SPLIT]:
            # Although the files end with ".json", they are actually JSONL files
            split_file_name: str = f"{'dev' if split == VALID_SPLIT else split}.json"
            split_path: str = os.path.join(data_path, split_file_name)

            with open(split_path, "r") as f:
                for line in f:
                    # The data fields and their explanations can be found here:
                    # https://github.com/MedMCQA/MedMCQA#data-fields
                    example: Dict[str, str] = json.loads(line.rstrip())

                    references: List[Reference] = [
                        # Value of "cop" corresponds to the index of the correct option
                        Reference(Output(text=example[option]), tags=[CORRECT_TAG] if index == example["cop"] else [])
                        for option, index in MedMCQAScenario.ANSWER_OPTION_TO_INDEX.items()
                    ]
                    instance: Instance = Instance(
                        input=Input(text=example["question"]),
                        references=references,
                        split=split,
                    )
                    instances.append(instance)

        return instances
