import csv
import os
from typing import List

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    TEST_SPLIT,
    Input,
)


_DATA_DIRRECTORY_PATH = "restricted/helpdesk_call_summarization/HELM Sample Transcripts_20241221_0045"


class HelpdeskCallSummarizationScenario(Scenario):
    """Helpdesk call summarization."""

    name = "helpdesk_call_summarization"
    description = "Helpdesk call summarization."
    tags = ["helpdesk_call_center"]

    def get_instances(self, output_path: str) -> List[Instance]:
        instances: List[Instance] = []
        for file_name in os.listdir(_DATA_DIRRECTORY_PATH):
            if not file_name.endswith(".csv") or not file_name.startswith("Call1-"):
                continue
            file_path = os.path.join(_DATA_DIRRECTORY_PATH, file_name)
            with open(file_path) as f:
                csv_reader = csv.reader(f)
                prompt_lines = [f"{row[0]}: {row[4]}" for row in csv_reader]
            prompt = "\n".join(prompt_lines)
            instance_id = file_name.removeprefix("Call1-").removesuffix(".csv")
            input = Input(text=prompt)
            instance = Instance(id=instance_id, input=input, references=[], split=TEST_SPLIT)
            instances.append(instance)
        return instances
