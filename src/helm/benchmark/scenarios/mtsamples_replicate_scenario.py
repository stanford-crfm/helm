import os
import requests
from typing import List
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    CORRECT_TAG,
    TEST_SPLIT,
    Input,
    Output,
)
from helm.common.general import ensure_directory_exists


class MTSamplesReplicateScenario(Scenario):
    """
    MTSamples.com is designed to give you access to a big collection of transcribed medical reports.
    These samples can be used by learning, as well as working medical transcriptionists for their daily
    transcription needs. We present the model with patient information and request it to generate a corresponding
    treatment plan.

    Sample Synthetic Prompt:
    Given various information about a patient, return a reasonable treatment plan for the patient.

    - Extracts `PLAN`, `SUMMARY`, or `FINDINGS` as the reference (PLAN preferred).
    - Removes `PLAN` from the input text but keeps other sections.
    - Ignores files that do not contain any of these reference sections.
    """

    GIT_HASH = "ebc104a4f96c5b7602242f301e081e9934a23344"
    API_BASE_URL = (
        f"https://api.github.com/repos/raulista1997/benchmarkdata/contents/mtsamples_processed?ref={GIT_HASH}"
    )
    RAW_BASE_URL = f"https://raw.githubusercontent.com/raulista1997/benchmarkdata/{GIT_HASH}/mtsamples_processed/"

    name = "mtsamples_replicate"
    description = (
        "A dataset of clinical notes where the model is prompted to generate "
        "a reasonable treatment plan for the patient based on transcribed medical reports."
    )
    tags = ["medical", "transcription", "plan_generation"]

    def fetch_file_list(self) -> List[str]:
        """
        Uses the GitHub API to fetch the list of `.txt` files at a specific commit.
        """
        response = requests.get(self.API_BASE_URL)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch file list from GitHub API: {response.text}")

        files = response.json()
        return [f["name"] for f in files if f["name"].endswith(".txt")]

    def download_file(self, file_name: str, output_dir: str) -> str:
        """
        Downloads a text file from GitHub and saves it locally.
        """
        file_url = self.RAW_BASE_URL + file_name
        file_path = os.path.join(output_dir, file_name)

        if not os.path.exists(file_path):
            response = requests.get(file_url)
            if response.status_code != 200:
                raise Exception(f"Failed to download {file_url}")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(response.text)

        return file_path

    def extract_sections(self, text: str) -> tuple:
        """
        Extracts `PLAN`, `SUMMARY`, and `FINDINGS` sections from the text.
        Returns (plan, summary, findings) as a tuple, ensuring uppercase detection.
        """
        plan, summary, findings = None, None, None
        text_upper = text.upper()

        if "PLAN:" in text_upper:
            plan = text.split("PLAN:")[1].split("\n", 1)[0].strip()

        if "SUMMARY:" in text_upper:
            summary = text.split("SUMMARY:")[1].split("\n", 1)[0].strip()

        if "FINDINGS:" in text_upper:
            findings = text.split("FINDINGS:")[1].split("\n", 1)[0].strip()

        return plan, summary, findings

    def remove_plan_section(self, text: str) -> str:
        """
        Removes `PLAN:` section from the input text while keeping the rest.
        """
        sections = ["PLAN:"]
        for section in sections:
            if section in text:
                text = text.split(section)[0].strip()  # Keep content before PLAN
        return text

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Downloads, processes, and converts MTSamples data into HELM format.
        """
        ensure_directory_exists(output_path)

        # Fetch list of available files from GitHub
        file_list = self.fetch_file_list()

        instances = []
        for file_name in file_list:
            try:
                # Download the text file
                file_path = self.download_file(file_name, output_path)

                # Read content
                with open(file_path, "r", encoding="utf-8") as f:
                    text_content = f.read().strip()

                # Extract structured sections
                plan, summary, findings = self.extract_sections(text_content)

                # Use plan > summary > findings as reference text
                reference_text = plan or summary or findings
                if not reference_text:
                    continue  # Ignore notes with no reference section

                # Remove PLAN section from input
                cleaned_text = self.remove_plan_section(text_content)

                # Create HELM instance
                instances.append(
                    Instance(
                        input=Input(text=cleaned_text),  # Processed text without PLAN
                        references=[Reference(Output(text=reference_text), tags=[CORRECT_TAG])],
                        split=TEST_SPLIT,
                    )
                )
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

        return instances
