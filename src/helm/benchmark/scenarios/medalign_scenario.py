from typing import List

from helm.benchmark.presentation.taxonomy_info import TaxonomyInfo
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TEST_SPLIT,
    CORRECT_TAG,
    PassageQuestionInput,
    Output,
    ScenarioMetadata,
)
from helm.benchmark.scenarios.medalign_scenario_helper import return_dataset_dataframe  # type: ignore


class MedalignScenario(Scenario):
    """
    Scenario defining the MedAlign task as defined in the following work by Fleming et al:
    @article{fleming2023medalign,
      title={MedAlign: A Clinician-Generated Dataset for Instruction Following with Electronic Medical Records},
      author={Scott L. Fleming
        and Alejandro Lozano
        and William J. Haberkorn
        and Jenelle A. Jindal
        and Eduardo P. Reis
        and Rahul Thapa
        and Louis Blankemeier
        and Julian Z. Genkins
        and Ethan Steinberg
        and Ashwin Nayak
        and Birju S. Patel
        and Chia-Chun Chiang
        and Alison Callahan
        and Zepeng Huo
        and Sergios Gatidis
        and Scott J. Adams
        and Oluseyi Fayanju
        and Shreya J. Shah
        and Thomas Savage
        and Ethan Goh
        and Akshay S. Chaudhari
        and Nima Aghaeepour
        and Christopher Sharp
        and Michael A. Pfeffer
        and Percy Liang
        and Jonathan H. Chen
        and Keith E. Morse
        and Emma P. Brunskill
        and Jason A. Fries
        and Nigam H. Shah},
      journal={arXiv preprint arXiv:2308.14089},
      year={2023}
    }
    Each instance includes:
    - input: the instruction and patient record
    - reference: the clinical 'gold standard' completion for the instruction for the given patient record
    This is a clinical instruction-following task, wherein a generative language model must follow
    the instructions using the provided patient record. As explained in the MedAlign work, each example
    is guaranteed to be completable for the given patient record.
    This task is evaluated using COMET and BERTScore metrics.
    """

    name = "medalign"
    description = (
        "MedAlign is a benchmark that evaluates a model's ability to interpret and follow"
        "instructions grounded in longitudinal electronic health records (EHR). Each instance"
        "includes an event-stream style patient record and a natural language question or task,"
        "requiring clinically informed reading comprehension and reasoning."
    )
    tags = ["knowledge", "reasoning", "biomedical"]

    def __init__(self, max_length: int, data_path: str):
        super().__init__()
        self.max_length = max_length
        self.data_path = data_path

    def process_tsv(self, data) -> List[Instance]:
        instances: List[Instance] = []
        for index, row in data.iterrows():
            question = row["prompt"]
            ground_truth_answer = row["clinician_response"]

            prompt = PassageQuestionInput(passage="", question=question)

            instance = Instance(
                input=prompt,
                references=[Reference(Output(text=ground_truth_answer), tags=[CORRECT_TAG])],
                split=TEST_SPLIT,
            )
            instances.append(instance)
        return instances

    def get_instances(self, output_path: str) -> List[Instance]:
        dataset = return_dataset_dataframe(self.max_length, self.data_path)
        return self.process_tsv(dataset)

    def get_metadata(self):
        return ScenarioMetadata(
            name="medalign",
            display_name="MedAlign",
            short_display_name="MedAlign",
            description="MedAlign is a benchmark that evaluates a model's ability to interpret and "
            "follow instructions grounded in longitudinal electronic health records (EHR). "
            "Each instance includes an event-stream style patient record and a natural "
            "language question or task, requiring clinically informed reading comprehension "
            "and reasoning [(Fleming et al., 2023)](https://arxiv.org/abs/2308.14089).",
            taxonomy=TaxonomyInfo(
                task="Text generation",
                what="Answer questions and follow instructions over longitudinal EHR",
                when="Any",
                who="Clinician, Researcher",
                language="English",
            ),
            main_metric="medalign_accuracy",
            main_split="test",
        )
