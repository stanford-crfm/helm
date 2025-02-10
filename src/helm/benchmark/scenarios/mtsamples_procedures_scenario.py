import os
import requests
import re
from bs4 import BeautifulSoup
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


class MTSamplesProceduresScenario(Scenario):
    """
    Processes the MTSamples dataset from raw text files.

    - Dynamically fetches the list of text files from GitHub.
    - Downloads and processes medical transcription files.
    - Extracts `PLAN`, `SUMMARY`, or `FINDINGS` sections as references.
    - Ensures these sections are excluded from the input text.
    - Filters out files that do not contain any of the three reference sections.

    Data source: https://github.com/raulista1997/benchmarkdata/tree/main/mtsample_procedure
    """

    GITHUB_DIR_URL = "https://github.com/raulista1997/benchmarkdata/tree/main/mtsample_procedure"
    RAW_BASE_URL = "https://raw.githubusercontent.com/raulista1997/benchmarkdata/refs/heads/main/mtsample_procedure/"

    name = "mtsamples"
    description = "Medical transcription dataset with structured plan generation."
    tags = ["medical", "transcription", "plan_generation"]

    def fetch_file_list(self) -> List[str]:
        """
        Scrapes the GitHub directory page to get a list of all `.txt` files.
        """
        response = requests.get(self.GITHUB_DIR_URL)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch file list from GitHub ({self.GITHUB_DIR_URL})")

        soup = BeautifulSoup(response.text, "html.parser")
        file_links = [
            link.text for link in soup.find_all("a", {"href": re.compile(r"/raulista1997/benchmarkdata/blob/main/mtsample_procedure/.*\.txt$")})
        ]
        return file_links

    def download_file(self, file_name: str, output_dir: str) -> str:
        """
        Downloads a text file from GitHub and saves it locally.
        """
        file_url = self.RAW_BASE_URL + file_name
        file_path = os.path.join(output_dir, file_name)

        if not os.path.exists(file_path):  # Avoid redundant downloads
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

    def remove_sections(self, text: str) -> str:
        """
        Removes `PLAN`, `SUMMARY`, and `FINDINGS` sections from the input text.
        """
        sections = ["PLAN:", "SUMMARY:", "FINDINGS:"]
        for section in sections:
            if section in text:
                text = text.split(section)[0].strip()  # Keep content before the section
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

                # Remove structured sections from input
                cleaned_text = self.remove_sections(text_content)

                # Create HELM instance
                instances.append(
                    Instance(
                        input=Input(text=cleaned_text),  # Processed text without sections
                        references=[Reference(Output(text=reference_text), tags=[CORRECT_TAG])],
                        split=TEST_SPLIT,
                    )
                )
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

        return instances
