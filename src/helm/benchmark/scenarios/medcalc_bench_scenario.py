from typing import Dict, List
from datasets import load_dataset

from helm.common.hierarchical_logger import hlog
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TEST_SPLIT,
    CORRECT_TAG,
    PassageQuestionInput,
    Output,
)


class MedCalcBenchScenario(Scenario):
    """
    MedCalc-Bench is the first medical calculation dataset used to benchmark
    LLMs ability to serve as clinical calculators.
    Each instance in the dataset consists of a patient note, a question asking
    to compute a specific clinical value, a final answer value, and a step-by-step
    solution explaining how the final answer was obtained. Our dataset covers 55
    different calculation tasks. We hope this dataset serves as a call to improve
    the verbal and computational reasoning skills of LLMs in medical settings.

    This dataset contains a training dataset of 10,053 instances and a testing
    dataset of 1,047 instances.

    Dataset: https://huggingface.co/datasets/ncbi/MedCalc-Bench-v1.0
    Paper: https://arxiv.org/abs/2406.12036

    Sample Prompt:
        Given a patient note and a clinical question, compute the requested medical value.
        Be as concise as possible.

        Patient note: A 70-year-old female was rushed into the ICU due to respiratory distress,
        following which she was promptly put on mechanical ventilation. Her delivered oxygen fell
        to 51 % FiO₂; meanwhile, her partial pressure of oxygen (PaO₂) registered at 74 mm Hg.
        She was conscious but visibly disoriented with a functional Glasgow Coma Score of 12.
        She was hypotensive with blood pressure of 91/70 mm Hg. Multiple vasopressors are being administered
        simultaneously including DOPamine at 4 mcg/kg/min, norEPINEPHrine at 0.06 mcg/kg/min,
        DOBUTamine at 3 mcg/kg/min, and EPINEPHrine at 0.03 mcg/kg/min. Laboratory evaluations
        revealed mild renal impairment with creatinine levels slightly elevated at 1.6 mg/dL
        and a bilirubin level of 1.9 mg/dL. Her platelet count was found to be 165,000/µL.
        Her daily urine output of 950 mL.
        Question: What is the patient's Sequential Organ Failure Assessment (SOFA) Score?

        Answer:

    @misc{khandekar2024medcalcbench,
        title={MedCalc-Bench: Evaluating Large Language Models for Medical Calculations},
        author={
            Nikhil Khandekar and Qiao Jin and Guangzhi Xiong and Soren Dunn and Serina S Applebaum and
            Zain Anwar and Maame Sarfo-Gyamfi and Conrad W Safranek and Abid A Anwar and Andrew Zhang and
            Aidan Gilson and Maxwell B Singer and Amisha Dave and Andrew Taylor and Aidong Zhang and
            Qingyu Chen and Zhiyong Lu
        },
        year={2024},
        eprint={2406.12036},
        archivePrefix={arXiv},
        primaryClass={
            id='cs.CL' full_name='Computation and Language' is_active=True alt_name='cmp-lg'
            in_archive='cs' is_general=False description='Covers natural language processing.
            Roughly includes material in ACM Subject Class I.2.7. Note that work on artificial
            languages (programming languages, logics, formal systems) that does not explicitly
            address natural-language issues broadly construed (natural-language processing, computational
            linguistics, speech, text retrieval, etc.) is not appropriate for this area.'
        }
    }
    """

    name = "medcalc_bench"
    description = (
        "A dataset which consists of a patient note, a question "
        "requesting to compute a specific medical value, and a ground truth answer."
    )
    tags = ["knowledge", "reasoning", "biomedical"]

    def __init__(self):
        super().__init__()

    def process_csv(self, data, split: str) -> List[Instance]:
        instances: List[Instance] = []
        hlog(f"Processing data for {split} split")
        for row in data:
            question = row["Question"]
            ground_truth_answer = row["Ground Truth Answer"]
            patient_note = row["Patient Note"]
            id = row["Row Number"]

            prompt = PassageQuestionInput(
                passage=patient_note + "\n", question=question + "\n", passage_prefix="Patient note: "
            )

            extra_data = {
                "category": row["Category"],
                "upper_limit": row["Upper Limit"],
                "lower_limit": row["Lower Limit"],
            }

            instance = Instance(
                input=prompt,
                references=[Reference(Output(text=ground_truth_answer), tags=[CORRECT_TAG])],
                extra_data=extra_data,
                split=split,
                id=id,
            )
            instances.append(instance)
        return instances

    def get_instances(self, output_path: str) -> List[Instance]:
        # Load the MedCalc-Bench dataset from Hugging Face
        dataset = load_dataset("ncbi/MedCalc-Bench-v1.0")

        # Process all the instances - limit to zero shot setting
        instances: List[Instance] = []
        splits: Dict[str, str] = {
            # "train": TRAIN_SPLIT,
            "test": TEST_SPLIT,
        }
        for hf_split, split in splits.items():
            data = dataset[hf_split]
            instances.extend(self.process_csv(data, split))

        return instances
