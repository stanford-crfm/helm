import os
from typing import List

from helm.common.general import ensure_directory_exists, ensure_file_downloaded
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    ALL_SPLITS,
    CORRECT_TAG,
    VALID_SPLIT,
    Input,
    Output,
)


class COVIDDialogScenario(Scenario):
    """
    From https://github.com/UCSD-AI4H/COVID-Dialogue, "COVID-Dialogue-Dataset-English is an English medical dialogue
    dataset about COVID-19 and other types of pneumonia. Patients who are concerned that they may be infected by
    COVID-19 or other pneumonia consult doctors and doctors provide advice. There are 603 consultations. Each
    consultation consists of ID, URL, Description of patient’s medical condition and Dialogue."

    The following is an example a patient-doctor interaction from the dataset:

    patient: i have all the symptoms except fever, i went to medicross and dr said i can get tested if i want to i'm
    not sure if i should. she gave me antibiotics klacid xl 500mg, she said i can take it if i feel worse i'm worried
    it will make immune system bad?

    in brief: antibiotic i don't recommend antibiotics for a simple viral upper respiratory tract infection unless
    examination revealed signs of acute bronchitis or sinusitis. they are not effective for viral infections like
    covid 19 with no bacterial lung involvement either. if you've been exposed to someone with covid 19 or or if you
    or someone you were exposed to travelled to a region where it was endemic, get tested would you like to video
    or text chat with me?

    @article{ju2020CovidDialog,
      title={CovidDialog: Medical Dialogue Datasets about COVID-19},
      author={Ju, Zeqian and Chakravorty, Subrato and He, Xuehai and Chen, Shu and Yang, Xingyi and Xie, Pengtao},
      journal={ https://github.com/UCSD-AI4H/COVID-Dialogue},
      year={2020}
    }
    """

    SOURCE_URL_TEMPLATE: str = (
        "https://worksheets.codalab.org/rest/bundles/0x6f1ac4b2e47043fcbb873b2af1c7ee0c/contents/blob/{file_name}"
    )

    name = "covid_dialog"
    description = "Medical dialogue dataset of conversations between doctors and patients on their COVID-19 concerns"
    tags = ["dialogue", "biomedical"]

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Build `Instance`s using the questions asked by concerned patients and the responses by doctors.
        """

        def download_and_read_lines(file_name: str) -> List[str]:
            file_path: str = os.path.join(data_path, file_name)
            ensure_file_downloaded(
                source_url=COVIDDialogScenario.SOURCE_URL_TEMPLATE.format(file_name=file_name),
                target_path=file_path,
                unpack=False,
            )

            with open(file_path) as f:
                return f.read().splitlines()

        data_path: str = os.path.join(output_path, "data")
        ensure_directory_exists(data_path)

        instances: List[Instance] = []
        for split in ALL_SPLITS:
            dataset_split: str = "val" if split == VALID_SPLIT else split

            # The files with the questions end with ".source"
            questions: List[str] = download_and_read_lines(f"{dataset_split}.source")

            # The files with the responses end with ".target"
            responses: List[str] = download_and_read_lines(f"{dataset_split}.target")

            for question, response in zip(questions, responses):
                # Questions in the dataset are already prepended with "patient:".
                # Remove those and add it via `input_prefix` of `AdapterSpec`.
                question = question.replace("patient: ", "")
                instances.append(
                    Instance(
                        input=Input(text=question),
                        references=[Reference(output=Output(text=response), tags=[CORRECT_TAG])],
                        split=split,
                    )
                )

        return instances
