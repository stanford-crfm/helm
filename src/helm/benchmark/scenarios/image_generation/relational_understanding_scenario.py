from typing import List, Set
import csv
import os

from helm.common.general import ensure_file_downloaded
from helm.benchmark.scenarios.scenario import Scenario, Instance, Input, TEST_SPLIT


class RelationalUnderstandingScenario(Scenario):
    """
    From Testing Relational Understanding in Text-Guided Image Generation, based on existing cognitive,
    linguistic, and developmental literature, the authors created a set of 15 relations (8 physical,
    7 agentic) and a set of 12 entities (6 objects, 6 agents). The physical relations were: in, on,
    under, covering, near, occluded by, hanging over, and tied to. The agentic relations were: pushing,
    pulling, touching, hitting, kicking, helping, and hindering. The objects were: box, cylinder,
    blanket, bowl, teacup, and knife. The agents were: man, woman, child, robot, monkey, and iguana.

    The authors created 5 different prompts for each relation, by randomly sampling two entities five
    times, resulting in 75 distinct basic relation prompts (e.g., a monkey touching an iguana). Withs
    these prompts, the authors showed that DALL-E 2 suffers from a significant lack of commonsense
    reasoning in the form of relational understanding.

    Paper: https://arxiv.org/abs/2208.00005
    Website: https://osf.io/sm68h
    """

    name = "relational_understanding"
    description = (
        "Consists of 75 basic relation prompts that tests commonsense reasoning "
        "([paper](https://arxiv.org/abs/2208.00005))."
    )
    tags = ["text-to-image", "reasoning"]

    def get_instances(self, output_path: str) -> List[Instance]:
        data_path: str = os.path.join(output_path, "choice_data.csv")
        ensure_file_downloaded(source_url="https://osf.io/download/tb3a4", target_path=data_path)

        instances: List[Instance] = []
        seen_prompts: Set[str] = set()
        with open(data_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            for i, row in enumerate(csv_reader):
                if i == 0:
                    # Skip the header
                    continue

                prompt: str = row[1]
                if prompt not in seen_prompts:
                    instances.append(Instance(Input(text=prompt), references=[], split=TEST_SPLIT))
                    seen_prompts.add(prompt)

        return instances
