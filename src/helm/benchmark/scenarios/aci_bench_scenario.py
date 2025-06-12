import json
import os
from typing import List
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    CORRECT_TAG,
    TRAIN_SPLIT,
    TEST_SPLIT,
    Input,
    Output,
)
from helm.common.general import ensure_file_downloaded


class ACIBenchScenario(Scenario):
    """
    From "Aci-bench: a Novel Ambient Clinical Intelligence Dataset for Benchmarking Automatic Visit Note Generation"
    (Yim et al.), ACI-Bench is the largest dataset to date tackling the problem of AI-assisted note generation from
    doctor-patient dialogue. This dataset enables benchmarking and evaluation of generative models, focusing on the
    arduous task of converting clinical dialogue into structured electronic medical records (EMR).

    Example from the dataset:

    Dialogue:
    [doctor] hi, brian. how are you?
    [patient] hi, good to see you.
    [doctor] it's good to see you too. so, i know the nurse told you a little bit about dax.
    [patient] mm-hmm.
    [doctor] i'd like to tell dax about you, okay?
    [patient] sure.

    Note:
    CHIEF COMPLAINT

    Follow-up of chronic problems.

    HISTORY OF PRESENT ILLNESS

    @Article{ACI-Bench,
    author = {Wen-wai Yim, Yujuan Fu, Asma Ben Abacha, Neal Snider, Thomas Lin, Meliha Yetisgen},
    title = {Aci-bench: a Novel Ambient Clinical Intelligence Dataset for Benchmarking Automatic Visit Note Generation},
    journal = {Nature Scientific Data},
    year = {2023},
    abstract = {Recent immense breakthroughs in generative models have precipitated re-imagined ubiquitous
    usage of these models in all applications. One area that can benefit by improvements in artificial intelligence (AI)
    is healthcare. The note generation task from doctor-patient encounters, and its associated electronic medical record
    documentation, is one of the most arduous time-consuming tasks for physicians. It is also a natural prime potential
    beneficiary to advances in generative models. However with such advances, benchmarking is more critical than ever.
    Whether studying model weaknesses or developing new evaluation metrics, shared open datasets are an imperative part
    of understanding the current state-of-the-art. Unfortunately as clinic encounter conversations are not routinely
    recorded and are difficult to ethically share due to patient confidentiality, there are no sufficiently large clinic
    dialogue-note datasets to benchmark this task. Here we present the Ambient Clinical Intelligence Benchmark
    corpus, the largest dataset to date tackling the problem of AI-assisted note generation from visit dialogue. We also
    present the benchmark performances of several common state-of-the-art approaches.}}

    Task:
    Given a doctor-patient dialogue, models must generate a clinical note that summarizes the conversation,
    focusing on the chief complaint, history of present illness, and other relevant clinical information.
    """

    PREFIX = (
        "https://raw.githubusercontent.com/"
        "wyim/aci-bench/e75b383172195414a7a68843ec4876e83e5409f7/data/challenge_data_json"
    )
    TRAIN_URL = f"{PREFIX}/train_full.json"
    TEST_URLS = [
        f"{PREFIX}/clinicalnlp_taskB_test1_full.json",
        f"{PREFIX}/clef_taskC_test3_full.json",
        f"{PREFIX}/clinicalnlp_taskC_test2_full.json",
    ]

    name = "aci_bench"
    description = "A dataset of patient-doctor conversations paired with structured clinical notes."
    tags = ["summarization", "medicine"]

    def download_json(self, url: str, output_path: str, file_name: str) -> str:
        """Download the JSON file and save it to the specified path."""
        json_path = os.path.join(output_path, file_name)
        ensure_file_downloaded(source_url=url, target_path=json_path, unpack=False)
        return json_path

    def process_json(self, json_path: str, split: str) -> List[Instance]:
        """Read and process the JSON file to generate instances."""
        instances: List[Instance] = []
        with open(json_path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)

            for entry in data["data"]:
                dialogue = entry["src"]
                note = entry["tgt"]

                # Prepare the input text (dialogue)
                input_text = f"Doctor-patient dialogue:\n\n{dialogue}"

                # Create an instance
                instance = Instance(
                    input=Input(text=input_text),
                    references=[Reference(Output(text=note), tags=[CORRECT_TAG])],
                    split=split,
                )
                instances.append(instance)

        return instances

    def get_instances(self, output_path: str) -> List[Instance]:
        """Download and process the dataset to generate instances."""
        instances: List[Instance] = []

        # Process training set
        train_json = self.download_json(self.TRAIN_URL, output_path, "aci_bench_train.json")
        instances.extend(self.process_json(train_json, TRAIN_SPLIT))

        # Process test sets
        for idx, test_url in enumerate(self.TEST_URLS, start=1):
            test_json = self.download_json(test_url, output_path, f"aci_bench_test_{idx}.json")
            instances.extend(self.process_json(test_json, TEST_SPLIT))

        return instances
