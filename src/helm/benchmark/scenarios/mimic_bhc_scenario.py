import json
from typing import Dict, List

from helm.benchmark.presentation.taxonomy_info import TaxonomyInfo
from helm.common.general import check_file_exists
from helm.benchmark.scenarios.scenario import (
    Input,
    Scenario,
    Instance,
    TEST_SPLIT,
    CORRECT_TAG,
    Reference,
    Output,
    ScenarioMetadata,
)


class MIMICBHCScenario(Scenario):
    r"""
    MIMIC-IV-BHC presents a curated collection of preprocessed discharge notes with labeled brief hospital
    course (BHC) summaries. This dataset is derived from MIMIC-IV (https://doi.org/10.1093/jamia/ocae312).

    In total, the dataset contains 270,033 clinical notes.
    The splits are provided by the dataset itself.

    Sample Synthetic Prompt:
        Summarize the clinical note into a brief hospital course.

        Clinical Note:
        <SEX> M <SERVICE> SURGERY <ALLERGIES> No Known Allergies \/ Adverse Drug Reactions
        ...
        continue to follow-up with your health care providers as an outpatient.

        Brief Hospital Course:
        Mr. ___ was pre-admitted on ___ for liver transplantation
        ...
        discharged home to continue home medications and follow-up as an outpatient.

    @article{aali2024dataset,
        title={A dataset and benchmark for hospital course summarization with adapted large language models},
        author={Aali, Asad and Van Veen, Dave and Arefeen, YI and Hom, Jason and Bluethgen, Christian
        and Reis, Eduardo Pontes and Gatidis, Sergios and Clifford, Namuun and Daws, Joseph
        and Tehrani, Arash and Kim, Jangwon and Chaudhari, Akshay},
        journal={Journal of the American Medical Informatics Association},
        volume={32},
        number={3},
        pages={470--479},
        year={2024},
        publisher={Oxford University Press}
    }

    @article{aali2024mimic,
        title={MIMIC-IV-Ext-BHC: Labeled Clinical Notes Dataset for Hospital Course Summarization},
        author={Aali, Asad and Van Veen, Dave and Arefeen, YI and Hom, Jason and Bluethgen, Christian
        and Reis, Eduardo Pontes and Gatidis, Sergios and Clifford, Namuun and Daws, Joseph
        and Tehrani, Arash and Kim, Jangwon and Chaudhari, Akshay},
        journal={PhysioNet},
        year={2024}
    }
    """

    name = "mimic_bhc"
    description = (
        "MIMIC-BHC is a benchmark focused on summarization of discharge notes into Brief"
        "Hospital Course (BHC) sections. It consists of curated discharge notes from MIMIC-IV,"
        "each paired with its corresponding BHC summary. The benchmark evaluates a model's"
        "ability to condense detailed clinical information into accurate, concise summaries that"
        "reflect the patient's hospital stay."
    )
    tags = ["summarization", "biomedical"]

    def __init__(self, data_path: str):
        super().__init__()
        self.data_path = data_path

    def get_instances(self, output_path: str) -> List[Instance]:
        check_file_exists(self.data_path, msg=f"[MIMICBHCScenario] Required data file not found: '{self.data_path}'")

        instances: List[Instance] = []
        # Limit to zero shot setting for now
        splits: Dict[str, str] = {
            # "train": TRAIN_SPLIT,
            # "validate": VALID_SPLIT,
            "test": TEST_SPLIT,
        }

        with open(self.data_path, "r") as f:
            data = [json.loads(line) for line in f]

        for data_split, split in splits.items():
            clinical_notes: List[str] = [x["input"] for x in data]
            bhc_summaries: List[str] = [x["target"] for x in data]
            assert len(clinical_notes) == len(bhc_summaries), "Notes and summaries must have the same length"
            for clinical_note, bhc_summary in zip(clinical_notes, bhc_summaries):
                if not clinical_note or not bhc_summary:
                    continue
                instances.append(
                    Instance(
                        input=Input(text=clinical_note),
                        references=[Reference(Output(text=bhc_summary), tags=[CORRECT_TAG])],
                        split=split,
                    )
                )

        return instances

    def get_metadata(self):
        return ScenarioMetadata(
            name="mimic_bhc",
            display_name="MIMIC-IV-BHC",
            short_display_name="MIMIC-BHC",
            description="MIMIC-BHC is a benchmark focused on summarization of discharge notes into "
            "Brief Hospital Course (BHC) sections. It consists of curated discharge notes "
            "from MIMIC-IV, each paired with its corresponding BHC summary. The benchmark "
            "evaluates a model's ability to condense detailed clinical information into "
            "accurate, concise summaries that reflect the patient's hospital stay [(Aali et "
            "al., 2024)](https://doi.org/10.1093/jamia/ocae312).",
            taxonomy=TaxonomyInfo(
                task="Text generation",
                what="Summarize the clinical note into a brief hospital course",
                when="Upon hospital discharge",
                who="Clinician",
                language="English",
            ),
            main_metric="mimic_bhc_accuracy",
            main_split="test",
        )
