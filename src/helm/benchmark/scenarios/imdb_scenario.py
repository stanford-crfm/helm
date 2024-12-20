import os
from typing import List, Dict, Optional

from helm.common.general import ensure_file_downloaded
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    CORRECT_TAG,
    TRAIN_SPLIT,
    VALID_SPLIT,
    Input,
    Output,
)
from helm.benchmark.scenarios.imdb_scenario_pinned_file_order import listdir_with_pinned_file_order


class IMDBScenario(Scenario):
    """
    The IMDb dataset is from the paper:
    https://ai.stanford.edu/~amaas/data/sentiment/

    IMDb is a text classification dataset containing 25,000 training reviews and 25,000 test reviews.
    Each sample contains a sentence with its corresponding sentiment (0: Negative, 1: Positive)

    We prompt models using the following format

        <passage>
        Sentiment:

        Target completion:
            <sentiment> (<sentiment>:Positive or Negative)

    Using an example from the training dataset, we have

    ```
    Very good drama although it appeared to have a few blank areas leaving the viewers
    to fill in the action for themselves.
    I can imagine life being this way for someone who can neither read nor write.
    This film simply smacked of the real world: the wife who is suddenly the sole supporter,
    the live-in relatives and their quarrels, the troubled child who gets knocked up and then,
    typically, drops out of school, a jackass husband who takes the nest egg and buys beer with it.
    2 thumbs up.
    Sentiment:
    Target completion:
        Positive
    ```

    The IMDB dataset has a contrast set, whose examples happen to be in the original train split. We thus
    assign all examples with valid contrast sets to the validation split, in addition to those
    from the original test set.
    """

    name = "imdb"
    description = "Sentiment analysis dataset."
    tags = ["sentiment_classification"]

    def __init__(self, only_contrast=False):
        """
        Args:
            only_contrast: Produce only inputs that have a contrast version.
        """
        super().__init__()
        self.only_contrast = only_contrast

    def get_split_instances(
        self, target_path: str, split: str, contrast_map: Dict[str, Dict[str, str]]
    ) -> List[Instance]:
        label_to_classname: Dict[str, str] = {"pos": "Positive", "neg": "Negative"}
        split_instances: List[Instance] = []
        split_to_subdirectory: Dict[str, str] = {TRAIN_SPLIT: "train", VALID_SPLIT: "test"}
        split_path: str = os.path.join(target_path, split_to_subdirectory[split])
        for class_name, label_id in label_to_classname.items():
            split_path_label: str = os.path.join(split_path, class_name)
            for file in listdir_with_pinned_file_order(target_path, split, class_name):
                with open(os.path.join(split_path_label, file), "r") as f:
                    context: str = f.read().strip()
                    prompt: str = context

                    instance_split: str = split
                    contrast_inputs: Optional[List[Input]] = None
                    contrast_references: Optional[List[List[Reference]]] = None

                    if context in contrast_map:
                        contrast_example: Dict[str, str] = contrast_map[context]
                        contrast_inputs = [Input(text=contrast_example["contrast_input"])]
                        contrast_references = [
                            [Reference(Output(text=contrast_example["contrast_answer"]), tags=[CORRECT_TAG])]
                        ]
                        instance_split = VALID_SPLIT  # we want to evaluate on contrast sets
                    elif self.only_contrast and split == VALID_SPLIT:
                        continue

                    instance: Instance = Instance(
                        input=Input(text=prompt),
                        references=[Reference(Output(text=label_id), tags=[CORRECT_TAG])],
                        split=instance_split,
                        contrast_inputs=contrast_inputs,
                        contrast_references=contrast_references,
                    )
                    split_instances.append(instance)
        return split_instances

    def get_contrast_map(self, target_path: str, mode: str) -> dict:
        # Download contrast sets
        label_name_to_id = {"Positive": "Positive", "Negative": "Negative"}
        orig_and_contrast_inputs: list = []

        for filename in [f"{mode}_original.tsv", f"{mode}_contrast.tsv"]:
            url = f"https://raw.githubusercontent.com/allenai/contrast-sets/main/IMDb/data/{filename}"
            target_path_current: str = os.path.join(target_path, filename)
            ensure_file_downloaded(source_url=url, target_path=target_path_current, unpack=False)
            with open(target_path_current, encoding="utf-8") as f:
                orig_and_contrast_inputs.append(f.readlines()[1:])

        contrast_map = {}

        for orig_line, contrast_line in zip(orig_and_contrast_inputs[0], orig_and_contrast_inputs[1]):
            orig_label_name, orig_context = orig_line.strip().split("\t")
            orig_label = label_name_to_id[orig_label_name]

            contrast_label_name, contrast_context = contrast_line.strip().split("\t")
            contrast_label = label_name_to_id[contrast_label_name]

            assert orig_context not in contrast_map
            contrast_map[orig_context] = {
                "original_answer": orig_label,
                "contrast_input": contrast_context,
                "contrast_answer": contrast_label,
            }
        return contrast_map

    def get_instances(self, output_path: str) -> List[Instance]:
        target_path: str = os.path.join(output_path, "data")

        instances: List[Instance] = []

        url: str = f'{"https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"}'
        ensure_file_downloaded(source_url=url, target_path=target_path, unpack=True)
        contrast_map = self.get_contrast_map(target_path, "test")
        contrast_map.update(self.get_contrast_map(target_path, "dev"))

        for split in [TRAIN_SPLIT, VALID_SPLIT]:
            instances.extend(self.get_split_instances(target_path, split, contrast_map))
        return instances
