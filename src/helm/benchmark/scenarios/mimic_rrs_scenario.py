import os
from typing import Dict, List

from helm.common.general import check_file_exists
from helm.benchmark.scenarios.scenario import (
    Input,
    Scenario,
    Instance,
    TEST_SPLIT,
    CORRECT_TAG,
    Reference,
    Output,
)


class MIMICRRSScenario(Scenario):
    """
    MIMIC-RRS is a biomedical question answering (QA) dataset collected from MIMIC-III and MIMIC-CXR
    radiology reports.
    In this scenario, we only consider the radiology reports from MIMIC-III.
    In total, the dataset contains 73,259 reports.
    The splits are provided by the dataset itself.

    Sample Synthetic Prompt:
        Generate the impressions of a radiology report based on its findings.

        Findings:
        The heart is normal in size. The lungs are clear.

        Impressions:

    @inproceedings{Chen_2023,
        title={Toward Expanding the Scope of Radiology Report Summarization to Multiple Anatomies and Modalities},
        url={http://dx.doi.org/10.18653/v1/2023.acl-short.41},
        DOI={10.18653/v1/2023.acl-short.41},
        booktitle={Proceedings of the 61st Annual Meeting of the Association
                   for Computational Linguistics (Volume 2: Short Papers)},
        publisher={Association for Computational Linguistics},
        author={Chen, Zhihong and Varma, Maya and Wan, Xiang and Langlotz, Curtis and Delbrouck, Jean-Benoit},
        year={2023},
        pages={469–484}
    }
    """

    name = "mimic_rrs"
    description = (
        "MIMIC-RRS is a benchmark constructed from radiology reports in the MIMIC-III"
        "database. It contains pairs of 'Findings' and 'Impression' sections, enabling evaluation"
        "of a model's ability to summarize diagnostic imaging observations into concise, clinically"
        "relevant conclusions."
    )
    tags = ["question_answering", "biomedical"]

    def __init__(self, data_path: str):
        super().__init__()
        self.data_path = data_path

    def get_instances(self, output_path: str) -> List[Instance]:
        instances: List[Instance] = []
        # Limit to zero shot setting for now
        splits: Dict[str, str] = {
            # "train": TRAIN_SPLIT,
            # "validate": VALID_SPLIT,
            "test": TEST_SPLIT,
        }

        for data_split, split in splits.items():
            split_findings_name: str = f"{data_split}.findings.tok"
            split_impressions_name: str = f"{data_split}.impression.tok"
            findings_path: str = os.path.join(self.data_path, split_findings_name)
            impressions_path: str = os.path.join(self.data_path, split_impressions_name)
            check_file_exists(
                findings_path, msg=f"[MIMICRRSScenario] Required findings file not found: '{findings_path}'"
            )
            check_file_exists(
                impressions_path, msg=f"[MIMICRRSScenario] Required impressions file not found: '{impressions_path}'"
            )
            findings: List[str] = self.read_file(findings_path)
            impressions: List[str] = self.read_file(impressions_path)
            assert len(findings) == len(impressions), "Findings and impressions must have the same length"
            for finding, impression in zip(findings, impressions):
                if not finding or not impression:
                    continue
                instances.append(
                    Instance(
                        input=Input(text=finding),
                        references=[Reference(Output(text=impression), tags=[CORRECT_TAG])],
                        split=split,
                    )
                )

        return instances

    def read_file(self, file_path: str) -> List[str]:
        with open(file_path, "r") as file:
            lines = file.readlines()
        lines = [line.strip() for line in lines]
        return lines
