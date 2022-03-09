import os
from typing import List, Dict

from common.general import ensure_file_downloaded
from .scenario import Scenario, Instance, Reference, TRAIN_SPLIT, CORRECT_TAG, TEST_SPLIT


class IMDbScenario(Scenario):
    """
    The IMDb dataset is from the paper:
        https://ai.stanford.edu/~amaas/data/sentiment/

    IMDb is a text classification dataset containing 25,000 training reviews and 25,000 test reviews.
    Each sample contains a sentence with its corresponding label (0: negative, 1: positive)

    We prompt models using the following format:
        <passage>

        Target completion:
            Label:<label> (<label>:positive or negative)

    Using an example from the training dataset, we have
    Very good drama although it appeared to have a few blank areas leaving the viewers
    to fill in the action for themselves.
    I can imagine life being this way for someone who can neither read nor write.
    This film simply smacked of the real world: the wife who is suddenly the sole supporter,
    the live-in relatives and their quarrels, the troubled child who gets knocked up and then,
    typically, drops out of school, a jackass husband who takes the nest egg and buys beer with it.
    2 thumbs up.

    Target completion:
        Label:positive

    """

    name = "imdb"
    description = "Sentiment analysis dataset."
    tags = ["sentiment_classification"]

    def __init__(self):
        pass

    def get_split_instances(self, split: str, path: str) -> List[Instance]:
        label_to_classname: Dict[str, str] = {"pos": "label:positive", "neg": "label:negative"}
        split_instances: List[Instance] = []
        for class_name, label_id in label_to_classname.items():
            split_path_label: str = os.path.join(path, class_name)
            all_file_name = os.listdir(split_path_label)
            for each_filename in all_file_name:
                sentence_path = os.path.join(split_path_label, each_filename)
                with open(sentence_path, "r") as f:
                    context = f.read().strip()
                    instance: Instance = Instance(
                        input=context + "\n", references=[Reference(output=label_id, tags=[CORRECT_TAG])], split=split
                    )
                    split_instances.append(instance)
        return split_instances

    def get_instances(self) -> List[Instance]:
        target_path: str = os.path.join(self.output_path, "data")

        instances: List[Instance] = []
        split_to_filename: Dict[str, str] = {TRAIN_SPLIT: "train", TEST_SPLIT: "test"}

        url: str = f'{"https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"}'
        ensure_file_downloaded(source_url=url, target_path=target_path, unpack=True)

        for split, filename in split_to_filename.items():
            split_path: str = os.path.join(target_path, filename)
            instances.extend(self.get_split_instances(split, split_path))
        return instances


class IMDbContrastSetScenario(IMDbScenario):
    """
    Contrast Sets for The IMDb dataset is from the paper:
        https://arxiv.org/abs/2004.02709

    Original repository can be found at:
        https://github.com/allenai/contrast-sets

    An example instance of a perturbation (from the original paper):
    Orginal instance: Hardly one to be faulted for his ambition or his vision,
    it is genuinely unexpected, then, to see all Park’s effort add up to so very little. . . .
    The premise is promising, gags are copious and offbeat humour
    abounds but it all fails miserably to create any meaningful connection with the audience.
    Label: negative

    Perturbed instance: Hardly one to be faulted for his ambition or his vision, here we see all Park’s effort come to
    fruition. . . . The premise is perfect, gags are hilarious and offbeat humour abounds, and it
    creates a deep connection with the audience.
    Label: positive
    """

    name = "imdb-contrast-sets"
    description = "Contrast Sets for IMDb text classification."
    tags = ["sentiment_classification", "robustness"]

    def __init__(self):
        pass

    def get_instances(self) -> List[Instance]:
        target_path: str = os.path.join(self.output_path, "data")

        # Note: Contrast Sets are constructed using the test set of some selected IMDb dataset.
        # Hence, it is safe to use training examples as the in-context examples.
        url: str = f'{"https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"}'
        ensure_file_downloaded(source_url=url, target_path=target_path, unpack=True)
        split_path: str = os.path.join(target_path, "train")
        training_instances: List[Instance] = self.get_split_instances(TRAIN_SPLIT, split_path)

        # Ensure contrast sets are downloaded
        # Note: We use the test set of IMDb dataset to construct the contrast set
        url = "https://raw.githubusercontent.com/allenai/contrast-sets/main/IMDb/data/test_contrast.tsv"
        target_path_constrast: str = os.path.join(target_path, "test_contrast.tsv")
        ensure_file_downloaded(source_url=url, target_path=target_path_constrast, unpack=False)

        contrast_instances: List[Instance] = []

        label_name_to_id = {"Positive": "label:positive", "Negative": "label:negative"}

        with open(target_path_constrast, encoding="utf-8") as f:
            # skip the head line
            f.readline()
            for eachline in f:
                label_name, perturbed_context = eachline.strip().split("\t")
                perturbed_label = label_name_to_id[label_name]
                instance: Instance = Instance(
                    input=perturbed_context + "\n",
                    references=[Reference(output=perturbed_label, tags=[CORRECT_TAG])],
                    split=TEST_SPLIT,
                )
                contrast_instances.append(instance)

        instances: List[Instance] = training_instances + contrast_instances
        return instances
