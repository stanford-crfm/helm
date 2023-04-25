from typing import List
import json

from helm.common.general import ensure_file_downloaded
import os
import csv
from .scenario import Scenario, Instance, Input, TEST_SPLIT, Reference, Output


class DetectionScenario(Scenario):
    """
    TODO: include description of the scenario. Link relevant papers.
    See https://github.com/stanford-crfm/helm/blob/vhelm/src/helm/benchmark/scenarios/draw_bench_scenario.py#L11
    as an example.
    """

    DATASET_DOWNLOAD_URL: str = (
        "https://docs.google.com/spreadsheets/d/1xwp0rUvec9iitc2z_Pi4XJ9J24mZbjZ1Gh0-KVWCOXc/"
        "gviz/tq?tqx=out:csv&sheet=detection_prompts_coco"
    )

    name = "detection"
    description = "TODO: include a short description of this scenario."
    tags = ["text-to-image"]

    def get_instances(self) -> List[Instance]:
        prompts_path: str = os.path.join(self.output_path, "prompts.csv")
        ensure_file_downloaded(source_url=self.DATASET_DOWNLOAD_URL, target_path=prompts_path)

        instances: List[Instance] = []

        with open(prompts_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            for i, row in enumerate(csv_reader):
                if i == 0:
                    # Skip the header
                    continue

                prompt: str = row[1]
                skill: str = row[0]

                # TODO parse from csv
                if skill == 'object':
                    references = {'object': 'person'}
                elif skill == 'count':
                    references = {'count': 4, 'object': 'person'}
                elif skill == 'spatial':
                    references = {'objects': ['person', 'vase'],
                                  'relation': 'left to'
                                  }

                instance = Instance(
                    Input(text=prompt),
                    references=[Reference(output=Output(text=json.dumps(references)), tags=[])],
                    split=TEST_SPLIT,
                    sub_split=skill,
                )
                instances.append(instance)

        return instances
