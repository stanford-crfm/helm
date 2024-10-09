import csv
import os
from typing import List

from helm.common.general import ensure_file_downloaded
from helm.benchmark.scenarios.scenario import Scenario, Instance, Input, TEST_SPLIT


class DrawBenchScenario(Scenario):
    """
    DrawBench is a comprehensive and challenging set of prompts that support the evaluation and comparison
    of text-to-image models. Across these 11 categories, DrawBench comprises 200 prompts in total.

    The 11 categories in DrawBench and the descriptions of each category are:

        1. Colors: Ability to generate objects with specified colors.
        2. Counting: Ability to generate specified number of objects.
        3. Conflicting: Ability to generate conflicting interactions between objects
        4. DALL-E: Subset of challenging prompts from
                   [Zero-Shot Text-to-Image Generation](https://arxiv.org/abs/2102.12092).
        5. Descriptions: Ability to understand complex and long text prompts describing objects.
        6. Gary Marcus et al. => Gary: Set of challenging prompts from
                                       [A very preliminary analysis of DALL-E 2](https://arxiv.org/abs/2204.13807).
        7. Misspellings: Ability to understand misspelled prompts.
        8. Positional: Ability to generate objects with specified spatial positioning.
        9. Rare Word => Rare: Ability to understand rare words.
        10. Reddit: Set of challenging prompts from DALL-E 2 Reddit.
        11. Text: Ability to generate quoted text.

    Setting parameter `category` to "all", returns instances with all the prompts.

    Paper: https://arxiv.org/abs/2205.11487
    """

    DATASET_DOWNLOAD_URL: str = (
        "https://docs.google.com/spreadsheets/d/1y7nAbmR4FREi6npB1u-Bo3GFdwdOPYJc617rBOxIRHY/"
        "gviz/tq?tqx=out:csv&sheet=Sheet1"
    )
    ALL_CATEGORY: str = "all"

    name = "draw_bench"
    description = (
        "A comprehensive and challenging benchmark for text-to-image models, used to evaluate Imagen "
        "([paper](https://arxiv.org/abs/2205.11487))."
    )
    tags = ["text-to-image"]

    def __init__(self, category: str):
        super().__init__()
        self.category: str = category

    def get_instances(self, output_path: str) -> List[Instance]:
        prompts_path: str = os.path.join(output_path, "prompts.csv")
        ensure_file_downloaded(source_url=self.DATASET_DOWNLOAD_URL, target_path=prompts_path)

        instances: List[Instance] = []
        with open(prompts_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            for i, row in enumerate(csv_reader):
                if i == 0:
                    # Skip the header
                    continue

                prompt: str = row[0]
                category: str = row[1]

                # Draw Bench does not have reference images
                instance = Instance(Input(text=prompt), references=[], split=TEST_SPLIT)

                # If all categories are requested, add all instances
                if category.startswith(self.category) or self.category == self.ALL_CATEGORY:
                    instances.append(instance)

        return instances
