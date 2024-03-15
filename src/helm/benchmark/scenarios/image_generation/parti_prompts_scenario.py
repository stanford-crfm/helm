import csv
import os
from typing import List

from helm.common.general import ensure_file_downloaded
from helm.benchmark.scenarios.scenario import Scenario, Instance, Input, TEST_SPLIT


class PartiPromptsScenario(Scenario):
    """
    PartiPrompts (P2) is a set of 1600 diverse English prompts that allow us to more comprehensively
    evaluate and test the limits of text-to-image synthesis models.

    Each prompt in the P2 benchmark is associated with two labels:
    1. Category: indicating a broad group that a prompt belongs to
    2. Challenge: highlighting an aspect which makes a prompt difficult

    Categories:
    - Abstract: Descriptions that represent abstract concepts, including single words and simple numbers.
    - World Knowledge: Descriptions focused on objects and places that exist in the real world.
    - People: Descriptions where the primary participants are human beings (but not specific individuals,
              living or dead).
    - Animals: Descriptions in which the primary participants are animals.
    - Illustrations: Descriptions of images that involve specific types of graphical representations,
                     including geometrical objects, diagrams, and symbols.
    - Artifacts: Descriptions that represent abstract concepts, including single words and simple numbers.
    - Food & Beverage: Descriptions of things animals, especially human beings, eat or drink.
    - Vehicles: Descriptions where the focus is on man-made devices for transportation.
    - Arts: Descriptions of existing paintings or intended to produce novel images in the format of a painting.
    - Indoor Scenes: Descriptions about objects and participants that occur indoors.
    - Outdoor Scenes: Descriptions about objects and participants that occur outdoors.
    - Produce & Plants: Descriptions focused on plants or their products (fruits, vegetables, seeds, etc).

    Challenges:
    - Simple Detail: Descriptions that include only simple or high-level details.
    - Fine-grained Detail: Descriptions that include very detailed specifications of attributes or
                           actions of entities or objects in a scene.
    - Complex: Descriptions that include many fine-grained, interacting details or relationships between multiple
               participants.
    - Quantity: Descriptions that specify particular counts of occurrences of subjects in a scene.
    - Style & Format: Descriptions that specifically focus on the visual manner in which a subject or scene
                      must be depicted.
    - Properties & Positioning: Descriptions that target precise assignment of properties to entities or
                                objects (often in the context of multiple entities or objects), and/or the
                                relative spatial arrangement of entities and objects with respect to one
                                another or landmarks in the scene.
    - Linguistic Structures: Long and/or abstract words or complex syntactic structures or semantic
                             ambiguities.
    - Writing & Symbols: Descriptions that require words or symbols to be accurately represented
                         in the context of the visual scene.
    - Imagination: Descriptions that include participants or interactions that are not, or are generally unlikely
                   to be, found in the modern day world.
    - Basic: Descriptions about a single subject or concept with little to no detail or embellishment.
    - Perspective: Descriptions that specify particular viewpoints or positioning of the subjects in a scene.

    Paper: https://arxiv.org/abs/2206.10789
    Website: https://parti.research.google/
    """

    DATASET_DOWNLOAD_URL: str = "https://raw.githubusercontent.com/google-research/parti/main/PartiPrompts.tsv"
    ALL_CATEGORY: str = "all"

    name = "parti_prompts"
    description = (
        "PartiPrompts (P2) is a set of 1600 diverse English prompts that allow to more comprehensively "
        "evaluate and test the limits of text-to-image synthesis models ([paper](https://arxiv.org/abs/2206.10789))."
    )
    tags = ["text-to-image"]

    def __init__(self, category: str):
        super().__init__()
        self.category: str = category

    def get_instances(self, output_path: str) -> List[Instance]:
        prompts_path: str = os.path.join(output_path, "prompts.tsv")
        ensure_file_downloaded(source_url=self.DATASET_DOWNLOAD_URL, target_path=prompts_path)

        instances: List[Instance] = []
        with open(prompts_path) as f:
            tsv_reader = csv.reader(f, delimiter="\t")
            for i, row in enumerate(tsv_reader):
                if i == 0:
                    # Skip the header
                    continue

                prompt: str = row[0]
                category: str = row[1]

                # P2 does not have reference images
                instance = Instance(Input(text=prompt), references=[], split=TEST_SPLIT)
                if category.startswith(self.category) or self.category == self.ALL_CATEGORY:
                    instances.append(instance)

        return instances
